from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ampere.neural.observer_dataset import ObserverSeries, ObserverWindowDataset, validate_window_cycles
from ampere.neural.observer_losses import (
    apply_observed_projection,
    decode_observer_prediction,
    observer_loss,
    observer_loss_weights,
    scheduled_teacher_lambda,
)
from ampere.neural.observer_models import ObserverModelConfig, make_observer_model
from ampere.neural.observer_target import average_observer_outputs, make_ema_target_observer, update_ema_target
from ampere.neural.utils import require_torch, resolve_device, set_random_seed


@dataclass(frozen=True)
class ObserverTrainConfig:
    model_type: str = "observer_gru"
    window_cycles: int = 8
    loss_variant: str = "observer_supervised"
    output_mode: str = "residual_dwell"
    normalization_mode: str = "branchwise"
    include_time_features: bool = True
    include_branch_static: bool = True
    use_branch_embeddings: bool = True
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.10
    branch_embedding_dim: int = 8
    gain_bound: float = 2.0
    transition_weight: float = 0.0
    gain_lambda: float = 0.0
    use_ema_target: bool = False
    ema_tau: float = 0.005
    teacher_lambda: float = 0.0
    teacher_mask: str = "unobserved_only"
    teacher_space: str = "residual_delta"
    teacher_warmup_epochs: int = 0
    teacher_ramp_epochs: int = 0
    teacher_perturbation: str = "none"
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
    def quick(cls, *, model_type: str = "observer_gru", seed: int = 42) -> "ObserverTrainConfig":
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
class ObserverTrainResult:
    model: Any
    ema_target_model: Any | None
    history: list[dict[str, float]]
    metadata: dict[str, Any]


def _make_loader(dataset, *, batch_size: int, shuffle: bool, seed: int):
    torch = require_torch()
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _batch_to_device(batch: dict[str, object], device: str) -> dict[str, object]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def _mean(rows: list[float]) -> float:
    return float(np.mean(rows)) if rows else float("nan")


def _gain_metrics(outputs: dict[str, object], batch: dict[str, object]) -> dict[str, float]:
    torch = require_torch()
    gain = outputs["gain"].detach()
    selected = torch.nn.functional.one_hot(batch["selected_branch_index"], num_classes=gain.shape[1]).to(
        dtype=gain.dtype,
        device=gain.device,
    )
    selected_gain = torch.sum(gain * selected, dim=1)
    off_gain = gain * (1.0 - selected)
    return {
        "selected_gain_mean": float(selected_gain.mean().item()),
        "off_gain_abs_mean": float(torch.abs(off_gain).mean().item()),
        "max_gain_abs": float(torch.abs(gain).max().item()),
    }


def _perturb_student_batch(
    batch: dict[str, object],
    series: ObserverSeries,
    *,
    mode: str,
) -> dict[str, object]:
    if mode == "none":
        return batch
    if mode != "light_mask_noise":
        raise ValueError(f"Unknown teacher perturbation mode: {mode}")
    torch = require_torch()
    x = batch["x"].clone()
    feature_index = {name: idx for idx, name in enumerate(series.dwell.feature_names)}
    selected_idx = feature_index.get("selected_branch_mask", feature_index.get("observed_mask"))
    if selected_idx is not None:
        non_selected = x[..., selected_idx] <= 0.5
    else:
        non_selected = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)

    last_idx = feature_index.get("last_observed_dwell_value_scaled")
    available_idx = feature_index.get("last_observed_available")
    since_idx = feature_index.get("time_since_last_seen_norm")
    if last_idx is not None:
        noise = torch.randn_like(x[..., last_idx]) * 0.01
        x[..., last_idx] = torch.where(non_selected, x[..., last_idx] + noise, x[..., last_idx])

    history_drop = (torch.rand(x.shape[:-1], device=x.device) < 0.05) & non_selected
    if last_idx is not None:
        x[..., last_idx] = torch.where(history_drop, torch.zeros_like(x[..., last_idx]), x[..., last_idx])
    if available_idx is not None:
        x[..., available_idx] = torch.where(history_drop, torch.zeros_like(x[..., available_idx]), x[..., available_idx])
    if since_idx is not None:
        x[..., since_idx] = torch.where(history_drop, torch.ones_like(x[..., since_idx]), x[..., since_idx])

    observed_value_idx = feature_index.get("observed_dwell_value_scaled")
    observed_mask_idx = feature_index.get("observed_mask")
    observed_drop = (torch.rand(x.shape[:-1], device=x.device) < 0.03) & non_selected
    if observed_value_idx is not None:
        x[..., observed_value_idx] = torch.where(
            observed_drop,
            torch.zeros_like(x[..., observed_value_idx]),
            x[..., observed_value_idx],
        )
    if observed_mask_idx is not None:
        x[..., observed_mask_idx] = torch.where(observed_drop, torch.zeros_like(x[..., observed_mask_idx]), x[..., observed_mask_idx])

    perturbed = dict(batch)
    perturbed["x"] = x
    return perturbed


def _eval_outputs(model, batch: dict[str, object], *, target_model=None, eval_mode: str = "student") -> dict[str, object]:
    if eval_mode == "student":
        return model(batch)
    if target_model is None:
        raise ValueError(f"Evaluation mode {eval_mode} requires an EMA target model")
    if eval_mode == "ema_teacher":
        return target_model(batch)
    if eval_mode == "student_teacher_average":
        student_outputs = model(batch)
        target_outputs = target_model(batch)
        return average_observer_outputs(student_outputs, target_outputs)
    raise ValueError(f"Unknown observer evaluation mode: {eval_mode}")


def _evaluate_model(
    model,
    loader,
    *,
    series: ObserverSeries,
    config: ObserverTrainConfig,
    device: str,
    target_model=None,
    eval_mode: str = "student",
) -> dict[str, float]:
    torch = require_torch()
    model.eval()
    if target_model is not None:
        target_model.eval()
    loss_rows: dict[str, list[float]] = {
        "total": [],
        "supervised": [],
        "observed": [],
        "energy": [],
        "transition": [],
        "gain": [],
        "teacher": [],
    }
    gain_rows: dict[str, list[float]] = {"selected_gain_mean": [], "off_gain_abs_mean": [], "max_gain_abs": []}
    abs_errors: list[float] = []
    observed_abs_errors: list[float] = []
    unobserved_abs_errors: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            outputs = _eval_outputs(model, batch, target_model=target_model, eval_mode=eval_mode)
            loss = observer_loss(
                outputs=outputs,
                batch=batch,
                series=series,
                variant=config.loss_variant,
                transition_weight=config.transition_weight,
                gain_lambda=config.gain_lambda,
                teacher_lambda=0.0,
                teacher_mask=config.teacher_mask,
                teacher_space=config.teacher_space,
                model_type=config.model_type,
            )
            pred_physical = decode_observer_prediction(outputs, series)
            pred_physical = apply_observed_projection(pred_physical, batch, enabled=config.hard_observed_projection)
            error = torch.abs(pred_physical - batch["target_physical"])
            mask = batch["observed_mask"] > 0.5
            abs_errors.append(float(error.mean().item()))
            if bool(mask.any()):
                observed_abs_errors.append(float(error[mask].mean().item()))
            if bool((~mask).any()):
                unobserved_abs_errors.append(float(error[~mask].mean().item()))
            for key in loss_rows:
                loss_rows[key].append(float(loss[key].item()))
            gains = _gain_metrics(outputs, batch)
            for key, value in gains.items():
                gain_rows[key].append(value)
    return {
        "total": _mean(loss_rows["total"]),
        "supervised": _mean(loss_rows["supervised"]),
        "observed": _mean(loss_rows["observed"]),
        "energy": _mean(loss_rows["energy"]),
        "transition": _mean(loss_rows["transition"]),
        "gain": _mean(loss_rows["gain"]),
        "teacher": _mean(loss_rows["teacher"]),
        "mae": _mean(abs_errors),
        "observed_mae": _mean(observed_abs_errors),
        "unobserved_mae": _mean(unobserved_abs_errors),
        "selected_gain_mean": _mean(gain_rows["selected_gain_mean"]),
        "off_gain_abs_mean": _mean(gain_rows["off_gain_abs_mean"]),
        "max_gain_abs": _mean(gain_rows["max_gain_abs"]),
    }


def make_model_for_series(series: ObserverSeries, config: ObserverTrainConfig):
    window_tokens = validate_window_cycles(series.dwell, config.window_cycles)
    model_config = ObserverModelConfig(
        model_type=config.model_type,
        branch_count=series.branch_count,
        feature_dim=series.feature_dim,
        window_tokens=window_tokens,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        branch_embedding_dim=config.branch_embedding_dim,
        use_branch_embeddings=config.use_branch_embeddings,
        gain_bound=config.gain_bound,
    )
    return make_observer_model(model_config), model_config


def observer_checkpoint_path(series: ObserverSeries, config: ObserverTrainConfig, checkpoint_dir: Path) -> Path:
    return checkpoint_dir / (
        f"{series.dataset_id}_{series.target_mode}_{config.model_type}_wc{config.window_cycles}_"
        f"{config.output_mode}_{config.loss_variant}_{config.normalization_mode}_"
        f"tw{config.transition_weight:g}_gl{config.gain_lambda:g}_"
        f"ema{int(config.use_ema_target)}_tau{config.ema_tau:g}_tl{config.teacher_lambda:g}_"
        f"tm{config.teacher_mask}_ts{config.teacher_space}_twarm{config.teacher_warmup_epochs}_"
        f"tramp{config.teacher_ramp_epochs}_tp{config.teacher_perturbation}_seed{config.seed}.pt"
    )


def train_observer_model(
    series: ObserverSeries,
    *,
    config: ObserverTrainConfig,
    checkpoint_dir: Path | None = None,
) -> ObserverTrainResult:
    torch = require_torch()
    set_random_seed(config.seed)
    device = resolve_device(config.device)
    use_amp = bool(config.mixed_precision and device.startswith("cuda"))
    model, model_config = make_model_for_series(series, config)
    model = model.to(device)
    target_model = make_ema_target_observer(model).to(device) if config.use_ema_target else None
    train_dataset = ObserverWindowDataset(
        series,
        split="train",
        window_cycles=config.window_cycles,
        max_windows=config.max_train_windows,
        seed=config.seed,
    )
    val_dataset = ObserverWindowDataset(
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
    best_target_state = None
    best_val_mae = float("inf")
    stale = 0
    accum = max(1, int(config.grad_accum_steps))
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_rows: dict[str, list[float]] = {
            "total": [],
            "supervised": [],
            "observed": [],
            "energy": [],
            "transition": [],
            "gain": [],
            "teacher": [],
        }
        optimizer.zero_grad(set_to_none=True)
        effective_teacher_lambda = scheduled_teacher_lambda(
            config.teacher_lambda if config.use_ema_target else 0.0,
            epoch=epoch,
            warmup_epochs=config.teacher_warmup_epochs,
            ramp_epochs=config.teacher_ramp_epochs,
        )
        for step, batch in enumerate(train_loader, start=1):
            batch = _batch_to_device(batch, device)
            perturbation_mode = config.teacher_perturbation if config.use_ema_target else "none"
            student_batch = _perturb_student_batch(batch, series, mode=perturbation_mode)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(student_batch)
                target_outputs = None
                if target_model is not None and effective_teacher_lambda > 0.0:
                    with torch.no_grad():
                        target_outputs = target_model(batch)
                loss = observer_loss(
                    outputs=outputs,
                    batch=batch,
                    series=series,
                    variant=config.loss_variant,
                    transition_weight=config.transition_weight,
                    gain_lambda=config.gain_lambda,
                    teacher_lambda=effective_teacher_lambda,
                    teacher_mask=config.teacher_mask,
                    teacher_space=config.teacher_space,
                    model_type=config.model_type,
                    target_outputs=target_outputs,
                )
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
                if target_model is not None:
                    update_ema_target(model, target_model, config.ema_tau)
        if len(train_loader) % accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if target_model is not None:
                update_ema_target(model, target_model, config.ema_tau)

        val = _evaluate_model(model, val_loader, series=series, config=config, device=device)
        row = {"epoch": float(epoch)}
        row.update({f"train_{key}": _mean(values) for key, values in train_loss_rows.items()})
        row.update({f"val_{key}": value for key, value in val.items()})
        row["teacher_lambda_effective"] = float(effective_teacher_lambda)
        history.append(row)
        if val["mae"] < best_val_mae - 1e-8:
            best_val_mae = val["mae"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_target_state = (
                {key: value.detach().cpu().clone() for key, value in target_model.state_dict().items()}
                if target_model is not None
                else None
            )
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if target_model is not None and best_target_state is not None:
        target_model.load_state_dict(best_target_state)

    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = observer_checkpoint_path(series, config, checkpoint_dir)
        torch.save(
            {
                "model_state": model.state_dict(),
                "ema_target_state": target_model.state_dict() if target_model is not None else None,
                "train_config": asdict(config),
                "model_config": asdict(model_config),
                "series": {
                    "dataset_id": series.dataset_id,
                    "target_mode": series.target_mode,
                    "reconstruction_mode": series.reconstruction_mode,
                    "output_mode": series.dwell.output_mode,
                    "normalization_mode": series.dwell.normalization_mode,
                    "feature_names": series.dwell.feature_names,
                    "branch_ids": series.dwell.branch_ids,
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
        "output_mode": series.dwell.output_mode,
        "normalization_mode": series.dwell.normalization_mode,
        "window_cycles": config.window_cycles,
        "window_tokens": int(validate_window_cycles(series.dwell, config.window_cycles)),
        "train_windows": int(len(train_dataset)),
        "val_windows": int(len(val_dataset)),
        "epochs_completed": int(len(history)),
        "best_val_mae": float(best_val_mae),
        "device": device,
        "mixed_precision_used": use_amp,
        "loss_weights": asdict(
            observer_loss_weights(
                config.loss_variant,
                transition_weight=config.transition_weight,
                gain_lambda=config.gain_lambda,
                teacher_lambda=config.teacher_lambda if config.use_ema_target else 0.0,
            )
        ),
        "transition_weight": float(config.transition_weight),
        "gain_lambda": float(config.gain_lambda),
        "use_ema_target": bool(config.use_ema_target),
        "ema_tau": float(config.ema_tau if config.use_ema_target else 0.0),
        "teacher_lambda": float(config.teacher_lambda if config.use_ema_target else 0.0),
        "teacher_mask": config.teacher_mask if config.use_ema_target else "none",
        "teacher_space": config.teacher_space if config.use_ema_target else "none",
        "teacher_warmup_epochs": int(config.teacher_warmup_epochs if config.use_ema_target else 0),
        "teacher_ramp_epochs": int(config.teacher_ramp_epochs if config.use_ema_target else 0),
        "teacher_perturbation": config.teacher_perturbation if config.use_ema_target else "none",
        "seed": int(config.seed),
        "use_branch_embeddings": config.use_branch_embeddings,
        "include_time_features": series.dwell.include_time_features,
        "include_branch_static": series.dwell.include_branch_static,
        "hard_observed_projection": config.hard_observed_projection,
        "gain_bound": config.gain_bound,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
    }
    return ObserverTrainResult(model=model, ema_target_model=target_model, history=history, metadata=metadata)


def load_observer_model_checkpoint(
    series: ObserverSeries,
    *,
    config: ObserverTrainConfig,
    checkpoint_path: Path,
) -> ObserverTrainResult:
    torch = require_torch()
    device = resolve_device(config.device)
    use_amp = bool(config.mixed_precision and device.startswith("cuda"))
    model, _model_config = make_model_for_series(series, config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    target_model = None
    if checkpoint.get("ema_target_state") is not None:
        target_model = make_ema_target_observer(model).to(device)
        target_model.load_state_dict(checkpoint["ema_target_state"])
        target_model.eval()

    train_dataset = ObserverWindowDataset(
        series,
        split="train",
        window_cycles=config.window_cycles,
        max_windows=config.max_train_windows,
        seed=config.seed,
    )
    val_dataset = ObserverWindowDataset(
        series,
        split="val",
        window_cycles=config.window_cycles,
        max_windows=config.max_val_windows,
        seed=config.seed + 1,
    )
    best_val_mae = float(checkpoint.get("best_val_mae", float("nan")))
    metadata = {
        "dataset_id": series.dataset_id,
        "target_mode": series.target_mode,
        "reconstruction_mode": series.reconstruction_mode,
        "model_type": config.model_type,
        "loss_variant": config.loss_variant,
        "output_mode": series.dwell.output_mode,
        "normalization_mode": series.dwell.normalization_mode,
        "window_cycles": config.window_cycles,
        "window_tokens": int(validate_window_cycles(series.dwell, config.window_cycles)),
        "train_windows": int(len(train_dataset)),
        "val_windows": int(len(val_dataset)),
        "epochs_completed": int(checkpoint.get("train_config", {}).get("epochs_completed", 0)),
        "best_val_mae": best_val_mae,
        "device": device,
        "mixed_precision_used": use_amp,
        "loss_weights": asdict(
            observer_loss_weights(
                config.loss_variant,
                transition_weight=config.transition_weight,
                gain_lambda=config.gain_lambda,
                teacher_lambda=config.teacher_lambda if config.use_ema_target else 0.0,
            )
        ),
        "transition_weight": float(config.transition_weight),
        "gain_lambda": float(config.gain_lambda),
        "use_ema_target": bool(config.use_ema_target),
        "ema_tau": float(config.ema_tau if config.use_ema_target else 0.0),
        "teacher_lambda": float(config.teacher_lambda if config.use_ema_target else 0.0),
        "teacher_mask": config.teacher_mask if config.use_ema_target else "none",
        "teacher_space": config.teacher_space if config.use_ema_target else "none",
        "teacher_warmup_epochs": int(config.teacher_warmup_epochs if config.use_ema_target else 0),
        "teacher_ramp_epochs": int(config.teacher_ramp_epochs if config.use_ema_target else 0),
        "teacher_perturbation": config.teacher_perturbation if config.use_ema_target else "none",
        "seed": int(config.seed),
        "use_branch_embeddings": config.use_branch_embeddings,
        "include_time_features": series.dwell.include_time_features,
        "include_branch_static": series.dwell.include_branch_static,
        "hard_observed_projection": config.hard_observed_projection,
        "gain_bound": config.gain_bound,
        "checkpoint_path": str(checkpoint_path),
    }
    return ObserverTrainResult(model=model, ema_target_model=target_model, history=[], metadata=metadata)


def predict_observer_matrix(
    model,
    series: ObserverSeries,
    *,
    config: ObserverTrainConfig,
    split: str = "test",
    target_model=None,
    eval_mode: str = "student",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    torch = require_torch()
    device = resolve_device(config.device)
    dataset = ObserverWindowDataset(series, split=split, window_cycles=config.window_cycles, max_windows=None, seed=config.seed)
    loader = _make_loader(dataset, batch_size=config.batch_size, shuffle=False, seed=config.seed)
    predictions = np.full_like(series.dwell.target_physical, np.nan, dtype=np.float64)
    priors = np.full_like(series.dwell.target_physical, np.nan, dtype=np.float64)
    gains = np.full_like(series.dwell.target_physical, np.nan, dtype=np.float64)
    innovations = np.full(len(series.dwell.dwell_id), np.nan, dtype=np.float64)
    end_positions: list[int] = []
    model.eval()
    if target_model is not None:
        target_model.eval()
    with torch.no_grad():
        for batch in loader:
            ends = batch["end_position"].cpu().numpy().astype(int)
            batch = _batch_to_device(batch, device)
            outputs = _eval_outputs(model, batch, target_model=target_model, eval_mode=eval_mode)
            pred_physical = decode_observer_prediction(outputs, series)
            pred_physical = apply_observed_projection(pred_physical, batch, enabled=config.hard_observed_projection)
            prior_physical = series.dwell.target_normalizer.inverse_torch(outputs["prior_scaled"])
            predictions[ends] = pred_physical.detach().cpu().numpy()
            priors[ends] = prior_physical.detach().cpu().numpy()
            gains[ends] = outputs["gain"].detach().cpu().numpy()
            innovations[ends] = outputs["innovation_scaled"].detach().cpu().numpy()
            end_positions.extend([int(value) for value in ends])
    return predictions, priors, gains, innovations, np.asarray(end_positions, dtype=np.int64)
