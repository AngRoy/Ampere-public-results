from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ampere.neural.datasets import (
    NeuralSeries,
    WindowedReconstructionDataset,
    window_starts_for_split,
)
from ampere.neural.evaluate import build_neural_prediction_frame
from ampere.neural.losses import physics_guided_loss, weights_for_variant
from ampere.neural.models import CausalTCNReconstructor
from ampere.neural.utils import resolve_device, require_torch, set_random_seed


@dataclass(frozen=True)
class NeuralTrainingConfig:
    window_size: int = 128
    train_stride: int = 64
    prediction_stride: int = 128
    batch_size: int = 32
    epochs: int = 40
    patience: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden_channels: int = 32
    layers: int = 3
    dropout: float = 0.05
    max_train_windows: int | None = None
    max_val_windows: int | None = None
    seed: int = 42
    device: str = "auto"
    hard_observed_projection: bool = True

    @classmethod
    def quick(cls, *, seed: int = 42) -> "NeuralTrainingConfig":
        return cls(
            window_size=96,
            train_stride=96,
            prediction_stride=96,
            batch_size=32,
            epochs=5,
            patience=3,
            learning_rate=1e-3,
            hidden_channels=24,
            layers=2,
            dropout=0.03,
            max_train_windows=96,
            max_val_windows=48,
            seed=seed,
        )


@dataclass(frozen=True)
class NeuralRunResult:
    predictions: Any
    history: list[dict[str, float]]
    metadata: dict[str, Any]


def _make_loader(dataset, *, batch_size: int, shuffle: bool, seed: int):
    torch = require_torch()
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _batch_to_device(batch: dict[str, object], device: str) -> dict[str, object]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def _mean_loss_dict(loss_rows: list[dict[str, float]]) -> dict[str, float]:
    if not loss_rows:
        return {}
    keys = sorted(loss_rows[0])
    return {key: float(np.mean([row[key] for row in loss_rows])) for key in keys}


def _evaluate_loss(model, loader, *, series: NeuralSeries, loss_variant: str, device: str) -> dict[str, float]:
    torch = require_torch()
    weights = weights_for_variant(
        variant=loss_variant,
        dataset_id=series.dataset_id,
        target_mode=series.target_mode,
    )
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
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
            rows.append({key: float(value.item()) for key, value in loss.items()})
    return _mean_loss_dict(rows)


def train_model(
    series: NeuralSeries,
    *,
    loss_variant: str,
    config: NeuralTrainingConfig,
    checkpoint_dir: Path | None = None,
) -> tuple[object, list[dict[str, float]], dict[str, Any]]:
    torch = require_torch()
    set_random_seed(config.seed)
    device = resolve_device(config.device)
    train_starts = window_starts_for_split(
        series,
        split="train",
        window_size=config.window_size,
        stride=config.train_stride,
        max_windows=config.max_train_windows,
        seed=config.seed,
    )
    val_starts = window_starts_for_split(
        series,
        split="val",
        window_size=config.window_size,
        stride=config.prediction_stride,
        max_windows=config.max_val_windows,
        seed=config.seed + 1,
    )
    train_dataset = WindowedReconstructionDataset(series, train_starts, config.window_size)
    val_dataset = WindowedReconstructionDataset(series, val_starts, config.window_size)
    train_loader = _make_loader(train_dataset, batch_size=config.batch_size, shuffle=True, seed=config.seed)
    val_loader = _make_loader(val_dataset, batch_size=config.batch_size, shuffle=False, seed=config.seed)

    use_calibration = loss_variant == "neural_full_valid"
    residual_feature = f"last_observed_value_b{series.branch_ids[0]:02d}"
    residual_start = series.feature_names.index(residual_feature) if residual_feature in series.feature_names else None
    model = CausalTCNReconstructor(
        input_channels=series.feature_count,
        branch_count=series.branch_count,
        hidden_channels=config.hidden_channels,
        layers=config.layers,
        dropout=config.dropout,
        use_calibration=use_calibration,
        residual_start_channel=residual_start,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    weights = weights_for_variant(
        variant=loss_variant,
        dataset_id=series.dataset_id,
        target_mode=series.target_mode,
    )

    best_state = None
    best_val = float("inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_rows = []
        for batch in train_loader:
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_rows.append({key: float(value.item()) for key, value in loss.items()})

        train_loss = _mean_loss_dict(train_rows)
        val_loss = _evaluate_loss(model, val_loader, series=series, loss_variant=loss_variant, device=device)
        epoch_row = {"epoch": float(epoch)}
        epoch_row.update({f"train_{key}": value for key, value in train_loss.items()})
        epoch_row.update({f"val_{key}": value for key, value in val_loss.items()})
        history.append(epoch_row)
        current_val = val_loss.get("total", float("inf"))
        if current_val < best_val - 1e-8:
            best_val = current_val
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{series.dataset_id}_{series.target_mode}_{series.reconstruction_mode}_{loss_variant}.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": asdict(config),
                "dataset_id": series.dataset_id,
                "target_mode": series.target_mode,
                "reconstruction_mode": series.reconstruction_mode,
                "loss_variant": loss_variant,
                "feature_names": series.feature_names,
                "branch_ids": series.branch_ids,
                "scaler": {"mean": series.scaler.mean, "std": series.scaler.std},
                "best_val_total": best_val,
            },
            checkpoint_path,
        )

    metadata = {
        "dataset_id": series.dataset_id,
        "target_mode": series.target_mode,
        "reconstruction_mode": series.reconstruction_mode,
        "loss_variant": loss_variant,
        "neural_architecture": "causal_tcn",
        "device": device,
        "torch_version": torch.__version__,
        "train_windows": int(len(train_starts)),
        "val_windows": int(len(val_starts)),
        "best_val_total": float(best_val),
        "epochs_completed": int(len(history)),
        "use_calibration": use_calibration,
        "residual_last_observed": residual_start is not None,
        "loss_weights": asdict(weights),
        "feature_names": series.feature_names,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
    }
    return model, history, metadata


def predict_matrix(
    model,
    series: NeuralSeries,
    *,
    config: NeuralTrainingConfig,
    split: str = "test",
) -> np.ndarray:
    torch = require_torch()
    device = resolve_device(config.device)
    starts = window_starts_for_split(
        series,
        split=split,
        window_size=config.window_size,
        stride=config.prediction_stride,
        max_windows=None,
        seed=config.seed,
    )
    dataset = WindowedReconstructionDataset(series, starts, config.window_size)
    loader = _make_loader(dataset, batch_size=config.batch_size, shuffle=False, seed=config.seed)
    pred_sum = np.zeros_like(series.target_physical, dtype=np.float64)
    pred_count = np.zeros_like(series.target_physical, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            starts_batch = batch["start"].cpu().numpy().astype(int)
            batch = _batch_to_device(batch, device)
            prediction = model(batch["x"]).detach().cpu().numpy()
            physical = series.scaler.inverse_numpy(np.transpose(prediction, (0, 2, 1)))
            for row_idx, start in enumerate(starts_batch):
                stop = start + config.window_size
                pred_sum[start:stop] += physical[row_idx]
                pred_count[start:stop] += 1.0

    covered = pred_count > 0
    output = np.full_like(series.target_physical, np.nan, dtype=np.float64)
    output[covered] = pred_sum[covered] / pred_count[covered]
    missing = (series.split == split) & ~np.isfinite(output).all(axis=1)
    if bool(np.any(missing)):
        output[missing] = series.scaler.mean
    return output


def train_and_predict(
    series: NeuralSeries,
    *,
    loss_variant: str,
    config: NeuralTrainingConfig,
    checkpoint_dir: Path | None = None,
) -> NeuralRunResult:
    model, history, metadata = train_model(
        series,
        loss_variant=loss_variant,
        config=config,
        checkpoint_dir=checkpoint_dir,
    )
    p_hat = predict_matrix(model, series, config=config, split="test")
    predictions = build_neural_prediction_frame(
        series,
        p_hat,
        loss_variant=loss_variant,
        split="test",
        hard_observed_projection=config.hard_observed_projection,
    )
    return NeuralRunResult(predictions=predictions, history=history, metadata=metadata)
