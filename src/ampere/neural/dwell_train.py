from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ampere.neural.dwell_dataset import DwellSeries, DwellWindowDataset, validate_window_cycles
from ampere.neural.dwell_losses import decode_dwell_prediction, dwell_loss_weights, dwellnet_loss
from ampere.neural.dwell_models import DwellModelConfig, make_dwell_model
from ampere.neural.utils import resolve_device, require_torch, set_random_seed


@dataclass(frozen=True)
class DwellTrainConfig:
    model_type: str = "dwell_mlp"
    window_cycles: int = 4
    loss_variant: str = "dwell_supervised"
    output_mode: str = "residual_dwell"
    normalization_mode: str = "branchwise"
    include_time_features: bool = True
    include_branch_static: bool = True
    use_branch_embeddings: bool = True
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.10
    branch_embedding_dim: int = 8
    transformer_heads: int = 4
    epochs: int = 80
    patience: int = 12
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_accum_steps: int = 1
    max_train_windows: int | None = None
    max_val_windows: int | None = None
    seed: int = 42
    device: str = "auto"
    mixed_precision: bool = True
    hard_observed_projection: bool = True

    @classmethod
    def quick(cls, *, model_type: str = "dwell_mlp", seed: int = 42) -> "DwellTrainConfig":
        return cls(
            model_type=model_type,
            epochs=5,
            patience=3,
            batch_size=64,
            hidden_dim=32,
            num_layers=1,
            max_train_windows=256,
            max_val_windows=128,
            seed=seed,
        )


@dataclass(frozen=True)
class DwellTrainResult:
    model: Any
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


def _mean(rows: list[float]) -> float:
    return float(np.mean(rows)) if rows else float("nan")


def _evaluate_model(model, loader, *, series: DwellSeries, config: DwellTrainConfig, device: str) -> dict[str, float]:
    torch = require_torch()
    model.eval()
    loss_rows: dict[str, list[float]] = {"total": [], "supervised": [], "energy": [], "observed": [], "nonnegative": []}
    abs_errors: list[float] = []
    observed_abs_errors: list[float] = []
    unobserved_abs_errors: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            prediction = model(batch["x"])
            loss = dwellnet_loss(prediction=prediction, batch=batch, series=series, variant=config.loss_variant)
            pred_physical = decode_dwell_prediction(prediction, batch, series)
            if config.hard_observed_projection:
                mask = batch["observed_mask"] > 0.5
                pred_physical = torch.where(mask, batch["observed_values"], pred_physical)
            error = torch.abs(pred_physical - batch["target_physical"])
            mask = batch["observed_mask"] > 0.5
            abs_errors.append(float(error.mean().item()))
            if bool(mask.any()):
                observed_abs_errors.append(float(error[mask].mean().item()))
            if bool((~mask).any()):
                unobserved_abs_errors.append(float(error[~mask].mean().item()))
            for key in loss_rows:
                loss_rows[key].append(float(loss[key].item()))
    return {
        "total": _mean(loss_rows["total"]),
        "supervised": _mean(loss_rows["supervised"]),
        "energy": _mean(loss_rows["energy"]),
        "observed": _mean(loss_rows["observed"]),
        "nonnegative": _mean(loss_rows["nonnegative"]),
        "mae": _mean(abs_errors),
        "observed_mae": _mean(observed_abs_errors),
        "unobserved_mae": _mean(unobserved_abs_errors),
    }


def make_model_for_series(series: DwellSeries, config: DwellTrainConfig):
    window_tokens = validate_window_cycles(series, config.window_cycles)
    model_config = DwellModelConfig(
        model_type=config.model_type,
        branch_count=series.branch_count,
        feature_dim=series.feature_dim,
        window_tokens=window_tokens,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        branch_embedding_dim=config.branch_embedding_dim,
        use_branch_embeddings=config.use_branch_embeddings,
        transformer_heads=config.transformer_heads,
    )
    return make_dwell_model(model_config), model_config


def train_dwell_model(
    series: DwellSeries,
    *,
    config: DwellTrainConfig,
    checkpoint_dir: Path | None = None,
) -> DwellTrainResult:
    torch = require_torch()
    set_random_seed(config.seed)
    device = resolve_device(config.device)
    use_amp = bool(config.mixed_precision and device.startswith("cuda"))
    model, model_config = make_model_for_series(series, config)
    model = model.to(device)
    train_dataset = DwellWindowDataset(
        series,
        split="train",
        window_cycles=config.window_cycles,
        max_windows=config.max_train_windows,
        seed=config.seed,
    )
    val_dataset = DwellWindowDataset(
        series,
        split="val",
        window_cycles=config.window_cycles,
        max_windows=config.max_val_windows,
        seed=config.seed + 1,
    )
    train_loader = _make_loader(train_dataset, batch_size=config.batch_size, shuffle=True, seed=config.seed)
    val_loader = _make_loader(val_dataset, batch_size=config.batch_size, shuffle=False, seed=config.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float]] = []
    best_state = None
    best_val_mae = float("inf")
    stale = 0
    accum = max(1, int(config.grad_accum_steps))
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_rows: dict[str, list[float]] = {"total": [], "supervised": [], "energy": [], "observed": [], "nonnegative": []}
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = _batch_to_device(batch, device)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                prediction = model(batch["x"])
                loss = dwellnet_loss(prediction=prediction, batch=batch, series=series, variant=config.loss_variant)
                total = loss["total"] / accum
            scaler.scale(total).backward()
            for key in train_loss_rows:
                train_loss_rows[key].append(float(loss[key].item()))
            if step % accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        if len(train_loader) % accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        val = _evaluate_model(model, val_loader, series=series, config=config, device=device)
        row = {"epoch": float(epoch)}
        row.update({f"train_{key}": _mean(values) for key, values in train_loss_rows.items()})
        row.update({f"val_{key}": value for key, value in val.items()})
        history.append(row)
        if val["mae"] < best_val_mae - 1e-8:
            best_val_mae = val["mae"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / (
            f"{series.dataset_id}_{series.target_mode}_{config.model_type}_wc{config.window_cycles}_"
            f"{series.output_mode}_{config.loss_variant}_{series.normalization_mode}.pt"
        )
        torch.save(
            {
                "model_state": model.state_dict(),
                "train_config": asdict(config),
                "model_config": asdict(model_config),
                "series": {
                    "dataset_id": series.dataset_id,
                    "target_mode": series.target_mode,
                    "reconstruction_mode": series.reconstruction_mode,
                    "output_mode": series.output_mode,
                    "normalization_mode": series.normalization_mode,
                    "feature_names": series.feature_names,
                    "branch_ids": series.branch_ids,
                },
                "best_val_mae": best_val_mae,
            },
            checkpoint_path,
        )

    metadata = {
        "dataset_id": series.dataset_id,
        "target_mode": series.target_mode,
        "reconstruction_mode": series.reconstruction_mode,
        "model_type": config.model_type,
        "loss_variant": config.loss_variant,
        "output_mode": series.output_mode,
        "normalization_mode": series.normalization_mode,
        "window_cycles": config.window_cycles,
        "window_tokens": int(validate_window_cycles(series, config.window_cycles)),
        "train_windows": int(len(train_dataset)),
        "val_windows": int(len(val_dataset)),
        "epochs_completed": int(len(history)),
        "best_val_mae": float(best_val_mae),
        "device": device,
        "mixed_precision_used": use_amp,
        "loss_weights": asdict(dwell_loss_weights(config.loss_variant, series)),
        "use_branch_embeddings": config.use_branch_embeddings,
        "include_time_features": series.include_time_features,
        "include_branch_static": series.include_branch_static,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
    }
    return DwellTrainResult(model=model, history=history, metadata=metadata)


def predict_dwell_matrix(
    model,
    series: DwellSeries,
    *,
    config: DwellTrainConfig,
    split: str = "test",
) -> tuple[np.ndarray, np.ndarray]:
    torch = require_torch()
    device = resolve_device(config.device)
    dataset = DwellWindowDataset(series, split=split, window_cycles=config.window_cycles, max_windows=None, seed=config.seed)
    loader = _make_loader(dataset, batch_size=config.batch_size, shuffle=False, seed=config.seed)
    predictions = np.full_like(series.target_physical, np.nan, dtype=np.float64)
    end_positions: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            ends = batch["end_position"].cpu().numpy().astype(int)
            batch = _batch_to_device(batch, device)
            prediction = model(batch["x"])
            pred_physical = decode_dwell_prediction(prediction, batch, series)
            if config.hard_observed_projection:
                mask = batch["observed_mask"] > 0.5
                pred_physical = torch.where(mask, batch["observed_values"], pred_physical)
            values = pred_physical.detach().cpu().numpy()
            predictions[ends] = values
            end_positions.extend([int(value) for value in ends])
    return predictions, np.asarray(end_positions, dtype=np.int64)
