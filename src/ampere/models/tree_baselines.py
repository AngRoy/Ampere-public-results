from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ampere.features.tabular import make_row_features, power_representation_for, target_values
from ampere.features.window import WINDOW_KEYS, make_window_features


TREE_MODEL_FAMILIES = ("random_forest", "extra_trees", "hist_gradient_boosting")
MODEL_LEVELS = ("dense_row", "window_dwell")


@dataclass(frozen=True)
class ModelSpec:
    model_family: str
    factory: Callable[[int], object]
    supports_feature_importance: bool


@dataclass(frozen=True)
class TreeBaselineResult:
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    metadata: dict[str, object]


def tree_model_specs() -> dict[str, ModelSpec]:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor

    return {
        "random_forest": ModelSpec(
            model_family="random_forest",
            factory=lambda seed: RandomForestRegressor(
                n_estimators=24,
                max_depth=16,
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=seed,
            ),
            supports_feature_importance=True,
        ),
        "extra_trees": ModelSpec(
            model_family="extra_trees",
            factory=lambda seed: ExtraTreesRegressor(
                n_estimators=24,
                max_depth=18,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=seed,
            ),
            supports_feature_importance=True,
        ),
        "hist_gradient_boosting": ModelSpec(
            model_family="hist_gradient_boosting",
            factory=lambda seed: HistGradientBoostingRegressor(
                max_iter=80,
                max_leaf_nodes=31,
                learning_rate=0.08,
                l2_regularization=0.0,
                random_state=seed,
            ),
            supports_feature_importance=False,
        ),
    }


def _sample_training_indices(mask: np.ndarray, max_train_rows: int | None, random_state: int) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if max_train_rows is None or len(indices) <= max_train_rows:
        return indices
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(indices, size=max_train_rows, replace=False))


def _fit_model(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    model_family: str,
    train_mask: np.ndarray,
    max_train_rows: int | None,
    random_state: int,
) -> object:
    specs = tree_model_specs()
    if model_family not in specs:
        raise ValueError(f"Unknown tree model family: {model_family}")
    train_indices = _sample_training_indices(train_mask, max_train_rows, random_state)
    valid = np.isfinite(y[train_indices])
    if not valid.any():
        raise ValueError("No finite training targets available")
    train_indices = train_indices[valid]
    model = specs[model_family].factory(random_state)
    model.fit(X.iloc[train_indices], y[train_indices])
    return model


def _importance_frame(
    model: object,
    feature_columns: list[str],
    *,
    dataset_id: str,
    model_family: str,
    model_level: str,
    reconstruction_mode: str,
    target_mode: str,
    method: str,
) -> pd.DataFrame:
    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        return pd.DataFrame()
    frame = pd.DataFrame(
        {
            "dataset_id": dataset_id,
            "method": method,
            "model_family": model_family,
            "model_level": model_level,
            "reconstruction_mode": reconstruction_mode,
            "target_mode": target_mode,
            "feature": feature_columns,
            "importance": np.asarray(importance, dtype=float),
        }
    )
    return frame.sort_values("importance", ascending=False, kind="mergesort").reset_index(drop=True)


def _prediction_frame_from_long(
    long_df: pd.DataFrame,
    *,
    p_hat: np.ndarray,
    model_family: str,
    model_level: str,
    reconstruction_mode: str,
    target_mode: str,
) -> pd.DataFrame:
    method = f"{model_level}_{model_family}"
    columns = [
        "dataset_id",
        "source_type",
        "time_s",
        "time_index",
        "branch_id",
        "branch_name",
        "P_raw_signed",
        "is_observed",
        "P_observed",
        "selected_branch",
        "dt_s",
        "split",
        "source_file",
    ]
    optional = [
        "load_type",
        "dwell_id",
        "scan_cycle_id",
        "dwell_position",
        "dwell_s_requested",
        "dwell_s_effective",
        "time_axis_confidence",
        "scaling_applied",
        "P_clipped_nonnegative",
    ]
    present = [column for column in columns + optional if column in long_df.columns]
    pred = long_df[present].copy()
    pred["P_true"] = target_values(long_df, target_mode)
    pred["power_representation"] = power_representation_for(target_mode)
    pred["method"] = method
    pred["model_family"] = model_family
    pred["model_level"] = model_level
    pred["reconstruction_mode"] = reconstruction_mode
    pred["target_mode"] = target_mode
    pred["P_hat"] = np.asarray(p_hat, dtype=float)

    if target_mode == "raw_signed_power":
        observed = pred["is_observed"].astype(bool) & pred["P_observed"].notna()
        pred.loc[observed, "P_hat"] = pred.loc[observed, "P_observed"]

    ordered = [
        "dataset_id",
        "source_type",
        "time_s",
        "time_index",
        "branch_id",
        "branch_name",
        "P_true",
        "P_raw_signed",
        "power_representation",
        "method",
        "model_family",
        "model_level",
        "reconstruction_mode",
        "target_mode",
        "P_hat",
        "is_observed",
        "P_observed",
        "selected_branch",
        "dt_s",
        "split",
        "source_file",
    ]
    ordered.extend([column for column in optional if column in pred.columns and column not in ordered])
    return pred[ordered]


def _run_dense_row(
    long_df: pd.DataFrame,
    *,
    dataset_id: str,
    model_family: str,
    reconstruction_mode: str,
    target_mode: str,
    predict_splits: set[str],
    max_train_rows: int | None,
    random_state: int,
) -> TreeBaselineResult:
    features = make_row_features(
        long_df,
        reconstruction_mode=reconstruction_mode,
        target_mode=target_mode,
    )
    train_mask = long_df["split"].eq("train").to_numpy()
    predict_mask = long_df["split"].isin(predict_splits).to_numpy()
    model = _fit_model(
        features.X,
        features.y,
        model_family=model_family,
        train_mask=train_mask,
        max_train_rows=max_train_rows,
        random_state=random_state,
    )
    p_hat = model.predict(features.X.loc[predict_mask])
    predictions = _prediction_frame_from_long(
        long_df.loc[predict_mask],
        p_hat=p_hat,
        model_family=model_family,
        model_level="dense_row",
        reconstruction_mode=reconstruction_mode,
        target_mode=target_mode,
    )
    method = f"dense_row_{model_family}"
    importance = _importance_frame(
        model,
        features.feature_columns,
        dataset_id=dataset_id,
        model_family=model_family,
        model_level="dense_row",
        reconstruction_mode=reconstruction_mode,
        target_mode=target_mode,
        method=method,
    )
    return TreeBaselineResult(
        predictions=predictions,
        feature_importance=importance,
        metadata={
            "dataset_id": dataset_id,
            "method": method,
            "model_family": model_family,
            "model_level": "dense_row",
            "reconstruction_mode": reconstruction_mode,
            "target_mode": target_mode,
            "feature_columns": features.feature_columns,
            "train_rows": int(train_mask.sum()),
            "train_rows_used": int(min(train_mask.sum(), max_train_rows or train_mask.sum())),
            "prediction_rows": int(predict_mask.sum()),
        },
    )


def _run_window_dwell(
    long_df: pd.DataFrame,
    *,
    dataset_id: str,
    model_family: str,
    reconstruction_mode: str,
    target_mode: str,
    predict_splits: set[str],
    max_train_rows: int | None,
    random_state: int,
) -> TreeBaselineResult:
    features = make_window_features(
        long_df,
        reconstruction_mode=reconstruction_mode,
        target_mode=target_mode,
    )
    window = features.window_frame.copy()
    train_mask = window["split"].eq("train").to_numpy()
    predict_mask = window["split"].isin(predict_splits).to_numpy()
    model = _fit_model(
        features.X,
        features.y,
        model_family=model_family,
        train_mask=train_mask,
        max_train_rows=max_train_rows,
        random_state=random_state,
    )
    window_pred = window.loc[predict_mask, WINDOW_KEYS].copy()
    window_pred["P_hat_window"] = model.predict(features.X.loc[predict_mask])
    predict_long = long_df[long_df["split"].isin(predict_splits)].merge(window_pred, on=WINDOW_KEYS, how="inner")
    predictions = _prediction_frame_from_long(
        predict_long,
        p_hat=predict_long["P_hat_window"].to_numpy(dtype=float),
        model_family=model_family,
        model_level="window_dwell",
        reconstruction_mode=reconstruction_mode,
        target_mode=target_mode,
    )
    method = f"window_dwell_{model_family}"
    importance = _importance_frame(
        model,
        features.feature_columns,
        dataset_id=dataset_id,
        model_family=model_family,
        model_level="window_dwell",
        reconstruction_mode=reconstruction_mode,
        target_mode=target_mode,
        method=method,
    )
    return TreeBaselineResult(
        predictions=predictions,
        feature_importance=importance,
        metadata={
            "dataset_id": dataset_id,
            "method": method,
            "model_family": model_family,
            "model_level": "window_dwell",
            "reconstruction_mode": reconstruction_mode,
            "target_mode": target_mode,
            "feature_columns": features.feature_columns,
            "train_windows": int(train_mask.sum()),
            "train_windows_used": int(min(train_mask.sum(), max_train_rows or train_mask.sum())),
            "prediction_windows": int(predict_mask.sum()),
            "prediction_rows": int(len(predictions)),
        },
    )


def run_tree_baseline(
    long_df: pd.DataFrame,
    *,
    dataset_id: str,
    model_family: str,
    model_level: str,
    reconstruction_mode: str,
    target_mode: str,
    predict_splits: set[str],
    max_train_rows: int | None = 60_000,
    random_state: int = 42,
) -> TreeBaselineResult:
    if model_level == "dense_row":
        return _run_dense_row(
            long_df,
            dataset_id=dataset_id,
            model_family=model_family,
            reconstruction_mode=reconstruction_mode,
            target_mode=target_mode,
            predict_splits=predict_splits,
            max_train_rows=max_train_rows,
            random_state=random_state,
        )
    if model_level == "window_dwell":
        return _run_window_dwell(
            long_df,
            dataset_id=dataset_id,
            model_family=model_family,
            reconstruction_mode=reconstruction_mode,
            target_mode=target_mode,
            predict_splits=predict_splits,
            max_train_rows=max_train_rows,
            random_state=random_state,
        )
    raise ValueError(f"Unknown model level: {model_level}")
