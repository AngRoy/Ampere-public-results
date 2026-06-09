from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ampere.evaluation.metrics import MetricTables, evaluate_predictions
from ampere.features.splits import add_time_block_split
from ampere.features.tabular import make_row_features, target_column_for
from ampere.neural.datasets import (
    NeuralSeries,
    WindowedReconstructionDataset,
    build_neural_series,
    window_starts_for_split,
)
from ampere.neural.evaluate import build_neural_prediction_frame
from ampere.neural.losses import LossWeights, physics_guided_loss
from ampere.neural.models import CausalTCNReconstructor
from ampere.neural.train import NeuralTrainingConfig, _batch_to_device, _make_loader, predict_matrix
from ampere.neural.utils import resolve_device, require_torch, set_random_seed


@dataclass(frozen=True)
class ResidualComparison:
    predictions: pd.DataFrame
    metrics: MetricTables
    zoh_comparison: dict[str, float | int | str | None]


@dataclass(frozen=True)
class TinyOverfitResult:
    history: pd.DataFrame
    initial_loss: float
    final_loss: float
    loss_ratio: float
    train_windows: int
    window_size: int


@dataclass(frozen=True)
class TabularMlpResult:
    predictions: pd.DataFrame
    metrics: MetricTables
    history: pd.DataFrame
    feature_columns: list[str]
    metadata: dict[str, Any]


def residual_prior_matrix(series: NeuralSeries) -> np.ndarray:
    """Return the causal no-learning prior: last observed value per branch/time."""

    matrix = (
        series.long.pivot(index="time_index", columns="branch_id", values="last_observed_value")
        .sort_index()
        .reindex(columns=series.branch_ids)
        .to_numpy(dtype=float)
    )
    fallback = np.where(np.isfinite(series.target_physical), series.target_physical, series.scaler.mean)
    return np.where(np.isfinite(matrix), matrix, fallback).astype(float)


def prediction_frame_from_matrix(
    series: NeuralSeries,
    matrix: np.ndarray,
    *,
    method: str,
    split: str = "test",
    neural_architecture: str = "diagnostic",
    loss_variant: str | None = None,
    hard_observed_projection: bool = True,
) -> pd.DataFrame:
    frame = build_neural_prediction_frame(
        series,
        matrix,
        loss_variant=loss_variant or method,
        split=split,
        neural_architecture=neural_architecture,
        hard_observed_projection=hard_observed_projection,
    )
    frame["method"] = method
    frame["loss_variant"] = loss_variant or method
    frame["neural_architecture"] = neural_architecture
    return frame


def _last_observed_residual_start(series: NeuralSeries) -> int | None:
    first = f"last_observed_value_b{series.branch_ids[0]:02d}"
    return series.feature_names.index(first) if first in series.feature_names else None


def make_zero_residual_tcn(
    series: NeuralSeries,
    *,
    hidden_channels: int = 16,
    layers: int = 1,
    dropout: float = 0.0,
) -> CausalTCNReconstructor:
    """Create a TCN whose learned correction is exactly zero before training."""

    torch = require_torch()
    model = CausalTCNReconstructor(
        input_channels=series.feature_count,
        branch_count=series.branch_count,
        hidden_channels=hidden_channels,
        layers=layers,
        dropout=dropout,
        use_calibration=False,
        residual_start_channel=_last_observed_residual_start(series),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


def zero_residual_predictions(
    series: NeuralSeries,
    *,
    config: NeuralTrainingConfig,
    split: str = "test",
) -> tuple[np.ndarray, pd.DataFrame, dict[str, float]]:
    model = make_zero_residual_tcn(
        series,
        hidden_channels=config.hidden_channels,
        layers=config.layers,
        dropout=0.0,
    ).to(resolve_device(config.device))
    matrix = predict_matrix(model, series, config=config, split=split)
    prior = residual_prior_matrix(series)
    mask = series.split == split
    diff = matrix[mask] - prior[mask]
    stats = {
        "max_abs_difference_vs_residual_prior": float(np.nanmax(np.abs(diff))),
        "mean_abs_difference_vs_residual_prior": float(np.nanmean(np.abs(diff))),
    }
    frame = prediction_frame_from_matrix(
        series,
        matrix,
        method="untrained_zero_residual_tcn",
        split=split,
        neural_architecture="causal_tcn_zero_residual",
        loss_variant="untrained_zero_residual_tcn",
        hard_observed_projection=True,
    )
    return matrix, frame, stats


def normalization_diagnostics(series: NeuralSeries) -> dict[str, float | str]:
    transformed = series.scaler.transform_numpy(series.target_physical)
    restored = series.scaler.inverse_numpy(transformed)
    error = np.abs(restored - series.target_physical)
    return {
        "normalization_scope": "global_train_target",
        "scaler_mean": float(series.scaler.mean),
        "scaler_std": float(series.scaler.std),
        "roundtrip_max_abs_error": float(np.nanmax(error)),
        "roundtrip_mean_abs_error": float(np.nanmean(error)),
    }


def target_scale_by_branch(series: NeuralSeries) -> pd.DataFrame:
    rows = []
    for pos, branch_id in enumerate(series.branch_ids):
        values = series.target_physical[:, pos]
        rows.append(
            {
                "branch_id": int(branch_id),
                "branch_name": series.branch_names[pos],
                "target_mean": float(np.nanmean(values)),
                "target_std": float(np.nanstd(values)),
                "target_min": float(np.nanmin(values)),
                "target_max": float(np.nanmax(values)),
            }
        )
    return pd.DataFrame(rows)


def feature_scale_diagnostics(series: NeuralSeries) -> pd.DataFrame:
    rows = []
    for idx, name in enumerate(series.feature_names):
        values = series.features[:, idx]
        rows.append(
            {
                "feature": name,
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
            }
        )
    return pd.DataFrame(rows)


def scan_cycle_info(series: NeuralSeries, *, quick_window_size: int = 96) -> dict[str, int | bool]:
    dwell_samples = int(np.nanmax(series.dwell_position) + 1)
    scan_cycle_samples = int(series.branch_count * dwell_samples)
    return {
        "branch_count": int(series.branch_count),
        "dwell_samples": dwell_samples,
        "scan_cycle_samples": scan_cycle_samples,
        "quick_window_size": int(quick_window_size),
        "quick_window_is_scan_multiple": bool(quick_window_size % scan_cycle_samples == 0),
        "quick_window_remainder": int(quick_window_size % scan_cycle_samples),
    }


def validate_scan_aligned_window(window_size: int, scan_cycle_samples: int) -> dict[str, int | bool]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if scan_cycle_samples <= 0:
        raise ValueError("scan_cycle_samples must be positive")
    return {
        "window_size": int(window_size),
        "scan_cycle_samples": int(scan_cycle_samples),
        "is_multiple": bool(window_size % scan_cycle_samples == 0),
        "remainder": int(window_size % scan_cycle_samples),
    }


def feature_parity_audit(long_df: pd.DataFrame, series: NeuralSeries) -> pd.DataFrame:
    tree = make_row_features(
        long_df,
        reconstruction_mode=series.reconstruction_mode,
        target_mode=series.target_mode,
    )
    useful_tree_features = [
        "time_s_norm",
        "time_index_norm",
        "scan_cycle_id_norm",
        "dwell_id_norm",
        "scan_cycle_mod_8",
        "dwell_id_mod_16",
        "load_type",
        "branch_name",
    ]
    rows = []
    neural_text = " ".join(series.feature_names)
    for feature in useful_tree_features:
        if feature in {"load_type", "branch_name"}:
            present_in_tree = any(column.startswith(f"{feature}_") for column in tree.feature_columns)
            present_in_neural = feature in neural_text
        else:
            present_in_tree = feature in tree.feature_columns
            present_in_neural = feature in series.feature_names
        rows.append(
            {
                "feature_family": feature,
                "present_in_tree": bool(present_in_tree),
                "present_in_neural": bool(present_in_neural),
                "diagnosis": "missing_from_neural" if present_in_tree and not present_in_neural else "available_or_not_used",
            }
        )
    return pd.DataFrame(rows)


def compare_residual_prior_to_zoh(
    residual_predictions: pd.DataFrame,
    *,
    classical_predictions_path: Path,
) -> dict[str, float | int | str | None]:
    if not classical_predictions_path.exists():
        return {"status": "missing_classical_predictions", "path": str(classical_predictions_path)}

    classical = pd.read_parquet(classical_predictions_path)
    if "split" not in classical.columns or classical["split"].isna().all():
        classical = add_time_block_split(classical)
    target_mask = (
        classical["target_mode"].eq("raw_signed_power")
        if "target_mode" in classical.columns
        else classical.get("power_representation", pd.Series("", index=classical.index)).eq("raw_signed_power")
    )
    zoh = classical[classical["method"].eq("zoh") & classical["split"].eq("test") & target_mask].copy()
    if zoh.empty:
        zoh = classical[classical["method"].eq("zoh") & classical["split"].eq("test")].copy()
    if zoh.empty:
        return {"status": "missing_zoh_rows", "path": str(classical_predictions_path)}

    left = residual_predictions[["time_index", "branch_id", "P_hat"]].rename(columns={"P_hat": "residual_prior"})
    right = zoh[["time_index", "branch_id", "P_hat"]].rename(columns={"P_hat": "zoh"})
    merged = left.merge(right, on=["time_index", "branch_id"], how="inner")
    if merged.empty:
        return {"status": "no_common_rows", "path": str(classical_predictions_path)}
    diff = merged["residual_prior"].to_numpy(dtype=float) - merged["zoh"].to_numpy(dtype=float)
    return {
        "status": "ok",
        "rows_compared": int(len(merged)),
        "max_abs_difference": float(np.max(np.abs(diff))),
        "mean_abs_difference": float(np.mean(np.abs(diff))),
    }


def evaluate_residual_prior(
    series: NeuralSeries,
    *,
    split: str = "test",
    classical_predictions_path: Path | None = None,
) -> ResidualComparison:
    matrix = residual_prior_matrix(series)
    predictions = prediction_frame_from_matrix(
        series,
        matrix,
        method="residual_prior_only",
        split=split,
        neural_architecture="no_learning_last_observed",
        loss_variant="residual_prior_only",
        hard_observed_projection=True,
    )
    metrics = evaluate_predictions(predictions)
    zoh_comparison: dict[str, float | int | str | None] = {"status": "not_requested"}
    if classical_predictions_path is not None:
        zoh_comparison = compare_residual_prior_to_zoh(
            predictions,
            classical_predictions_path=classical_predictions_path,
        )
    return ResidualComparison(predictions=predictions, metrics=metrics, zoh_comparison=zoh_comparison)


def tiny_overfit_supervised(
    series: NeuralSeries,
    *,
    window_size: int,
    train_windows: int = 8,
    epochs: int = 80,
    learning_rate: float = 5e-3,
    hidden_channels: int = 32,
    layers: int = 2,
    seed: int = 42,
) -> TinyOverfitResult:
    torch = require_torch()
    set_random_seed(seed)
    device = resolve_device("auto")
    starts = window_starts_for_split(
        series,
        split="train",
        window_size=window_size,
        stride=window_size,
        max_windows=None,
        seed=seed,
    )[:train_windows]
    dataset = WindowedReconstructionDataset(series, starts, window_size)
    loader = _make_loader(dataset, batch_size=max(1, len(starts)), shuffle=False, seed=seed)
    model = CausalTCNReconstructor(
        input_channels=series.feature_count,
        branch_count=series.branch_count,
        hidden_channels=hidden_channels,
        layers=layers,
        dropout=0.0,
        use_calibration=False,
        residual_start_channel=_last_observed_residual_start(series),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    weights = LossWeights()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["x"])
            loss = physics_guided_loss(
                prediction=prediction,
                target=batch["target"],
                observed_values=batch["observed_values"],
                observed_mask=batch["observed_mask"],
                dt_s=batch["dt_s"],
                dwell_id=batch["dwell_id"],
                scaler=series.scaler,
                weights=weights,
                calibration_regularization=model.calibration_regularization,
            )
            loss["total"].backward()
            optimizer.step()
            losses.append(float(loss["total"].item()))
        history.append({"epoch": epoch, "train_total": float(np.mean(losses))})

    history_df = pd.DataFrame(history)
    initial = float(history_df["train_total"].iloc[0])
    final = float(history_df["train_total"].iloc[-1])
    ratio = final / initial if initial > 0 else 0.0
    return TinyOverfitResult(
        history=history_df,
        initial_loss=initial,
        final_loss=final,
        loss_ratio=float(ratio),
        train_windows=int(len(starts)),
        window_size=int(window_size),
    )


def _prediction_frame_from_long_rows(
    rows: pd.DataFrame,
    *,
    p_hat: np.ndarray,
    method: str,
    target_mode: str,
    reconstruction_mode: str,
    loss_variant: str,
    neural_architecture: str,
) -> pd.DataFrame:
    target_column = target_column_for(target_mode)
    frame = rows.copy()
    frame["P_true"] = frame[target_column].to_numpy(dtype=float)
    frame["P_hat"] = p_hat.astype(float)
    frame["power_representation"] = target_mode
    frame["method"] = method
    frame["model_family"] = "neural"
    frame["neural_architecture"] = neural_architecture
    frame["model_level"] = "dense_row"
    frame["reconstruction_mode"] = reconstruction_mode
    frame["target_mode"] = target_mode
    frame["loss_variant"] = loss_variant
    if target_mode == "raw_signed_power":
        observed = frame["is_observed"].astype(bool) & frame["P_observed"].notna()
        frame.loc[observed, "P_hat"] = frame.loc[observed, "P_observed"]
    columns = [
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
        "neural_architecture",
        "model_level",
        "reconstruction_mode",
        "target_mode",
        "loss_variant",
        "P_hat",
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
    return frame[columns + [column for column in optional if column in frame.columns]]


def train_tabular_mlp_feature_parity(
    long_df: pd.DataFrame,
    *,
    target_mode: str,
    reconstruction_mode: str = "online_safe",
    max_train_rows: int = 12000,
    max_val_rows: int = 4000,
    epochs: int = 20,
    batch_size: int = 512,
    learning_rate: float = 2e-3,
    seed: int = 42,
) -> TabularMlpResult:
    torch = require_torch()
    set_random_seed(seed)
    df = long_df.copy()
    if "split" not in df.columns:
        df = add_time_block_split(df)
    features = make_row_features(df, reconstruction_mode=reconstruction_mode, target_mode=target_mode)
    X = features.X.to_numpy(dtype=np.float32)
    y = features.y.astype(np.float32)
    split = df["split"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)

    train_idx = np.flatnonzero(split == "train")
    val_idx = np.flatnonzero(split == "val")
    test_idx = np.flatnonzero(split == "test")
    if len(train_idx) > max_train_rows:
        train_idx = np.sort(rng.choice(train_idx, size=max_train_rows, replace=False))
    if len(val_idx) > max_val_rows:
        val_idx = np.sort(rng.choice(val_idx, size=max_val_rows, replace=False))

    x_mean = X[train_idx].mean(axis=0)
    x_std = X[train_idx].std(axis=0)
    x_std = np.where(x_std > 1e-8, x_std, 1.0)
    y_mean = float(np.mean(y[train_idx]))
    y_std = float(np.std(y[train_idx]))
    if not np.isfinite(y_std) or y_std < 1e-8:
        y_std = 1.0

    def scale_x(values: np.ndarray) -> np.ndarray:
        return ((values - x_mean) / x_std).astype(np.float32)

    def scale_y(values: np.ndarray) -> np.ndarray:
        return ((values - y_mean) / y_std).astype(np.float32)

    device = resolve_device("auto")
    train_dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(scale_x(X[train_idx]), dtype=torch.float32),
        torch.as_tensor(scale_y(y[train_idx]).reshape(-1, 1), dtype=torch.float32),
    )
    val_x = torch.as_tensor(scale_x(X[val_idx]), dtype=torch.float32).to(device)
    val_y = torch.as_tensor(scale_y(y[val_idx]).reshape(-1, 1), dtype=torch.float32).to(device)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_fn = torch.nn.SmoothL1Loss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).item()) if len(val_idx) else float("nan")
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss})

    model.eval()
    preds = np.empty(len(df), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(test_idx), 8192):
            idx = test_idx[start : start + 8192]
            xb = torch.as_tensor(scale_x(X[idx]), dtype=torch.float32).to(device)
            pred_scaled = model(xb).detach().cpu().numpy().reshape(-1)
            preds[idx] = pred_scaled * y_std + y_mean

    prediction_frame = _prediction_frame_from_long_rows(
        df.iloc[test_idx].copy(),
        p_hat=preds[test_idx],
        method="tabular_mlp_feature_parity",
        target_mode=target_mode,
        reconstruction_mode=reconstruction_mode,
        loss_variant="feature_parity_supervised",
        neural_architecture="tabular_mlp_feature_parity",
    )
    metrics = evaluate_predictions(prediction_frame)
    metadata = {
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "epochs": int(epochs),
        "feature_count": int(X.shape[1]),
        "target_mean": y_mean,
        "target_std": y_std,
    }
    return TabularMlpResult(
        predictions=prediction_frame,
        metrics=metrics,
        history=pd.DataFrame(history),
        feature_columns=features.feature_columns,
        metadata=metadata,
    )


def make_synthetic_alignment_long(
    *,
    time_steps: int = 72,
    branch_count: int = 4,
    dwell_samples: int = 3,
) -> pd.DataFrame:
    rows = []
    time_index = np.arange(time_steps)
    for t in time_index:
        selected = int((t // dwell_samples) % branch_count) + 1
        dwell_id = int(t // dwell_samples)
        scan_cycle = int(t // (dwell_samples * branch_count))
        dwell_position = int(t % dwell_samples)
        base_values = {
            1: 10.0,
            2: 20.0,
            3: float(t),
            4: float(30.0 + np.sin(t / 3.0)),
        }
        values = {
            branch_id: base_values.get(branch_id, float(40.0 + 5.0 * branch_id + np.sin(t / (branch_id + 1.0))))
            for branch_id in range(1, branch_count + 1)
        }
        for branch_id in range(1, branch_count + 1):
            value = values[branch_id]
            is_observed = branch_id == selected
            rows.append(
                {
                    "dataset_id": "synthetic_alignment",
                    "source_type": "synthetic",
                    "time_s": float(t),
                    "time_index": int(t),
                    "branch_id": int(branch_id),
                    "branch_name": f"Branch{branch_id:02d}",
                    "load_type": "synthetic",
                    "P_raw_signed": value,
                    "P_true": value,
                    "P_dwell_mean": value,
                    "power_representation": "raw_signed_power",
                    "P_observed": value if is_observed else np.nan,
                    "is_observed": bool(is_observed),
                    "selected_branch": selected,
                    "dwell_id": dwell_id,
                    "scan_cycle_id": scan_cycle,
                    "dwell_position": dwell_position,
                    "time_since_last_seen": 0.0 if is_observed else float(t + 1),
                    "time_to_next_seen": np.nan,
                    "last_observed_value": value if is_observed else values[branch_id],
                    "next_observed_value": np.nan,
                    "dt_s": 1.0,
                    "time_axis_confidence": "synthetic",
                    "scaling_applied": "none",
                    "source_file": "synthetic",
                    "dwell_s_requested": float(dwell_samples),
                    "dwell_s_effective": float(dwell_samples),
                }
            )
    return add_time_block_split(pd.DataFrame(rows))
