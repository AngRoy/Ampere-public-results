from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


RECONSTRUCTION_MODES = ("online_safe", "offline")
TARGET_MODES = ("raw_signed_power", "dwell_mean_power")
FUTURE_FEATURES = {"next_observed_value", "time_to_next_seen", "next_observed_available"}
PROHIBITED_EXACT_FEATURES = {
    "P_true",
    "P_raw_signed",
    "P_dwell_mean",
    "P_active_consumption",
    "P_clipped_nonnegative",
    "root_power",
}


@dataclass(frozen=True)
class FeatureMatrix:
    X: pd.DataFrame
    y: np.ndarray
    feature_columns: list[str]
    target_column: str
    target_mode: str
    reconstruction_mode: str


def target_column_for(target_mode: str) -> str:
    if target_mode == "raw_signed_power":
        return "P_true"
    if target_mode == "dwell_mean_power":
        return "P_dwell_mean"
    raise ValueError(f"Unknown target mode: {target_mode}")


def target_values(df: pd.DataFrame, target_mode: str) -> np.ndarray:
    column = target_column_for(target_mode)
    if column not in df.columns:
        raise ValueError(f"Target mode {target_mode} requires column {column}")
    return df[column].to_numpy(dtype=float)


def power_representation_for(target_mode: str) -> str:
    if target_mode not in TARGET_MODES:
        raise ValueError(f"Unknown target mode: {target_mode}")
    return target_mode


def _numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").astype(float)


def _safe_normalize(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    min_value = float(values.min())
    max_value = float(values.max())
    span = max_value - min_value
    if not np.isfinite(span) or span <= 0:
        return pd.Series(0.0, index=series.index, dtype=float)
    return (values - min_value) / span


def assert_no_feature_leakage(feature_columns: list[str], reconstruction_mode: str) -> None:
    exact = set(feature_columns) & PROHIBITED_EXACT_FEATURES
    if exact:
        raise ValueError(f"Feature matrix contains target/source leakage columns: {sorted(exact)}")
    if "P_raw_signed" in feature_columns or "P_true" in feature_columns:
        raise ValueError("Feature matrix contains direct truth leakage")
    if reconstruction_mode == "online_safe":
        future = set(feature_columns) & FUTURE_FEATURES
        if future:
            raise ValueError(f"Online-safe features contain future-derived columns: {sorted(future)}")


def make_feature_frame(
    df: pd.DataFrame,
    *,
    reconstruction_mode: str,
    fit_columns: list[str] | None = None,
) -> pd.DataFrame:
    if reconstruction_mode not in RECONSTRUCTION_MODES:
        raise ValueError(f"Unknown reconstruction mode: {reconstruction_mode}")

    features = pd.DataFrame(index=df.index)
    features["branch_id"] = _numeric(df, "branch_id")
    features["selected_branch"] = _numeric(df, "selected_branch")
    features["is_observed"] = _numeric(df, "is_observed")
    features["time_s_norm"] = _safe_normalize(_numeric(df, "time_s"))
    features["time_index_norm"] = _safe_normalize(_numeric(df, "time_index"))
    features["dwell_id_norm"] = _safe_normalize(_numeric(df, "dwell_id"))
    features["scan_cycle_id_norm"] = _safe_normalize(_numeric(df, "scan_cycle_id"))
    features["dwell_position"] = _numeric(df, "dwell_position")
    dwell_effective = _numeric(df, "dwell_s_effective", 1.0).replace(0, np.nan)
    dt_s = _numeric(df, "dt_s", 1.0)
    features["dwell_position_norm"] = (features["dwell_position"] * dt_s / dwell_effective).fillna(0.0)
    features["scan_cycle_mod_8"] = (_numeric(df, "scan_cycle_id") % 8).astype(float)
    features["dwell_id_mod_16"] = (_numeric(df, "dwell_id") % 16).astype(float)

    observed = _numeric(df, "P_observed")
    features["P_observed_filled"] = observed.fillna(0.0)
    features["P_observed_available"] = observed.notna().astype(float)
    last_observed = _numeric(df, "last_observed_value")
    features["last_observed_value"] = last_observed.fillna(0.0)
    features["last_observed_available"] = last_observed.notna().astype(float)
    features["time_since_last_seen"] = _numeric(df, "time_since_last_seen").fillna(1e9)

    if reconstruction_mode == "offline":
        next_observed = _numeric(df, "next_observed_value")
        features["next_observed_value"] = next_observed.fillna(0.0)
        features["next_observed_available"] = next_observed.notna().astype(float)
        features["time_to_next_seen"] = _numeric(df, "time_to_next_seen").fillna(1e9)

    categorical = [column for column in ["source_type", "load_type", "branch_name"] if column in df.columns]
    if categorical:
        dummies = pd.get_dummies(df[categorical].fillna("unknown"), columns=categorical, dtype=float)
        features = pd.concat([features, dummies], axis=1)

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if fit_columns is not None:
        features = features.reindex(columns=fit_columns, fill_value=0.0)
    assert_no_feature_leakage(list(features.columns), reconstruction_mode)
    return features


def make_row_features(
    df: pd.DataFrame,
    *,
    reconstruction_mode: str,
    target_mode: str,
    fit_columns: list[str] | None = None,
) -> FeatureMatrix:
    if target_mode not in TARGET_MODES:
        raise ValueError(f"Unknown target mode: {target_mode}")
    X = make_feature_frame(df, reconstruction_mode=reconstruction_mode, fit_columns=fit_columns)
    y = target_values(df, target_mode)
    return FeatureMatrix(
        X=X,
        y=y,
        feature_columns=list(X.columns),
        target_column=target_column_for(target_mode),
        target_mode=target_mode,
        reconstruction_mode=reconstruction_mode,
    )
