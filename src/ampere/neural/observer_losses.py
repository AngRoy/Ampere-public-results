from __future__ import annotations

from dataclasses import dataclass

from ampere.neural.dwell_losses import dwell_weighted_energy_loss, observed_dwell_consistency_loss
from ampere.neural.observer_dataset import OBSERVER_LOSS_VARIANTS, ObserverSeries
from ampere.neural.utils import require_torch

torch = require_torch()
import torch.nn.functional as F


OBSERVER_TEACHER_MASKS = ("all", "unobserved_only", "stale_unobserved_weighted")
OBSERVER_TEACHER_SPACES = ("absolute", "residual_delta")


@dataclass(frozen=True)
class ObserverLossWeights:
    supervised: float = 1.0
    observed: float = 0.0
    energy: float = 0.0
    transition: float = 0.0
    gain: float = 0.0
    teacher: float = 0.0


def observer_loss_weights(
    variant: str,
    *,
    transition_weight: float | None = None,
    gain_lambda: float | None = None,
    teacher_lambda: float | None = None,
) -> ObserverLossWeights:
    if variant not in OBSERVER_LOSS_VARIANTS:
        raise ValueError(f"Unknown DwellObserver loss variant: {variant}")
    transition = 0.0 if transition_weight is None else float(transition_weight)
    gain = 0.0 if gain_lambda is None else float(gain_lambda)
    teacher = 0.0 if teacher_lambda is None else float(teacher_lambda)
    if variant == "observer_supervised":
        return ObserverLossWeights(gain=gain, teacher=teacher)
    if variant == "observer_transition":
        return ObserverLossWeights(transition=transition if transition > 0.0 else 0.50, gain=gain, teacher=teacher)
    if variant == "observer_energy":
        return ObserverLossWeights(energy=0.02, gain=gain, teacher=teacher)
    if variant == "observer_gain_regularized":
        return ObserverLossWeights(gain=gain if gain > 0.0 else 0.01, teacher=teacher)
    raise ValueError(f"Unknown DwellObserver loss variant: {variant}")


def decode_observer_prediction(outputs: dict[str, object], series: ObserverSeries):
    return series.dwell.target_normalizer.inverse_torch(outputs["posterior_scaled"])


def apply_observed_projection(pred_physical, batch: dict[str, object], *, enabled: bool = True):
    if not enabled:
        return pred_physical
    mask = batch["observed_mask"] > 0.5
    return torch.where(mask, batch["observed_values"], pred_physical)


def observer_transition_loss(outputs: dict[str, object], batch: dict[str, object]):
    per_branch = F.smooth_l1_loss(outputs["posterior_scaled"], batch["target_scaled"], reduction="none")
    mask = batch["transition_mask"] > 0.5
    if not bool(mask.any()):
        return per_branch.sum() * 0.0
    return per_branch[mask].mean()


def observer_gain_regularization(outputs: dict[str, object], batch: dict[str, object]):
    gain = outputs["gain"]
    selected = torch.nn.functional.one_hot(
        batch["selected_branch_index"],
        num_classes=gain.shape[1],
    ).to(dtype=gain.dtype, device=gain.device)
    off_branch = gain * (1.0 - selected)
    selected_gain = torch.sum(gain * selected, dim=1)
    off_penalty = torch.mean(torch.abs(off_branch))
    selected_penalty = F.smooth_l1_loss(selected_gain, torch.ones_like(selected_gain))
    return off_penalty + 0.10 * selected_penalty


def observer_gain_regularization_allowed(model_type: str) -> bool:
    return model_type in {
        "observer_gru",
        "observer_mlp",
        "observer_mlp_learned_selected_gain",
        "observer_mlp_sparse_offbranch_gain",
    }


def scheduled_teacher_lambda(
    teacher_lambda: float,
    *,
    epoch: int,
    warmup_epochs: int = 0,
    ramp_epochs: int = 0,
) -> float:
    if teacher_lambda <= 0.0:
        return 0.0
    epoch = int(epoch)
    warmup_epochs = max(0, int(warmup_epochs))
    ramp_epochs = max(0, int(ramp_epochs))
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return float(teacher_lambda)
    ramp_step = min(1.0, max(0.0, (epoch - warmup_epochs) / float(ramp_epochs)))
    return float(teacher_lambda) * ramp_step


def observer_teacher_values(outputs: dict[str, object], batch: dict[str, object] | None, *, teacher_space: str):
    if teacher_space not in OBSERVER_TEACHER_SPACES:
        raise ValueError(f"Unknown teacher consistency space: {teacher_space}")
    if teacher_space == "absolute":
        return outputs["posterior_scaled"]
    if batch is None:
        raise ValueError("residual_delta teacher consistency requires batch['pre_base_scaled']")
    return outputs["posterior_scaled"] - batch["pre_base_scaled"]


def observer_teacher_mask_weights(batch: dict[str, object], *, teacher_mask: str):
    if teacher_mask not in OBSERVER_TEACHER_MASKS:
        raise ValueError(f"Unknown teacher consistency mask: {teacher_mask}")
    observed_mask = batch["observed_mask"].to(dtype=torch.float32)
    if teacher_mask == "all":
        return torch.ones_like(observed_mask)
    unobserved = (observed_mask <= 0.5).to(dtype=torch.float32)
    if teacher_mask == "unobserved_only":
        return unobserved

    stale = torch.clamp(batch["pre_time_since_last_seen_norm"].to(dtype=torch.float32), min=0.0, max=5.0)
    weights = unobserved * (1.0 + stale)
    active_count = unobserved.sum(dim=1, keepdim=True).clamp_min(1.0)
    active_mean = (weights.sum(dim=1, keepdim=True) / active_count).clamp_min(1e-6)
    return torch.where(unobserved > 0.0, weights / active_mean, torch.zeros_like(weights))


def observer_teacher_consistency_loss(
    outputs: dict[str, object],
    target_outputs: dict[str, object] | None,
    *,
    batch: dict[str, object] | None = None,
    teacher_mask: str = "all",
    teacher_space: str = "absolute",
):
    if target_outputs is None:
        return outputs["posterior_scaled"].sum() * 0.0
    student_prediction = observer_teacher_values(outputs, batch, teacher_space=teacher_space)
    target_prediction = observer_teacher_values(target_outputs, batch, teacher_space=teacher_space).detach()
    per_branch = F.smooth_l1_loss(student_prediction, target_prediction, reduction="none")
    if batch is None:
        return per_branch.mean()
    weights = observer_teacher_mask_weights(batch, teacher_mask=teacher_mask).to(dtype=per_branch.dtype, device=per_branch.device)
    denom = weights.sum()
    if float(denom.detach().cpu()) <= 0.0:
        return per_branch.sum() * 0.0
    return torch.sum(per_branch * weights) / denom


def mix_observer_outputs(first: dict[str, object], second: dict[str, object]) -> dict[str, object]:
    mixed: dict[str, object] = {}
    for key, value in first.items():
        other = second.get(key)
        if hasattr(value, "shape") and hasattr(other, "shape") and value.shape == other.shape:
            mixed[key] = 0.5 * (value + other)
        else:
            mixed[key] = value
    return mixed


def observer_loss(
    *,
    outputs: dict[str, object],
    batch: dict[str, object],
    series: ObserverSeries,
    variant: str,
    transition_weight: float | None = None,
    gain_lambda: float | None = None,
    teacher_lambda: float | None = None,
    teacher_mask: str = "all",
    teacher_space: str = "absolute",
    model_type: str | None = None,
    target_outputs: dict[str, object] | None = None,
) -> dict[str, object]:
    if gain_lambda and model_type is not None and not observer_gain_regularization_allowed(model_type):
        gain_lambda = 0.0
    weights = observer_loss_weights(
        variant,
        transition_weight=transition_weight,
        gain_lambda=gain_lambda,
        teacher_lambda=teacher_lambda,
    )
    pred_physical = decode_observer_prediction(outputs, series)
    supervised = F.smooth_l1_loss(outputs["posterior_scaled"], batch["target_scaled"])
    observed = observed_dwell_consistency_loss(pred_physical, batch["observed_values"], batch["observed_mask"])
    energy = dwell_weighted_energy_loss(pred_physical, batch["target_physical"], batch["dwell_s_effective"])
    transition = observer_transition_loss(outputs, batch)
    gain = observer_gain_regularization(outputs, batch)
    teacher = observer_teacher_consistency_loss(
        outputs,
        target_outputs,
        batch=batch,
        teacher_mask=teacher_mask,
        teacher_space=teacher_space,
    )
    total = (
        weights.supervised * supervised
        + weights.observed * observed
        + weights.energy * energy
        + weights.transition * transition
        + weights.gain * gain
        + weights.teacher * teacher
    )
    return {
        "total": total,
        "supervised": supervised.detach(),
        "observed": observed.detach(),
        "energy": energy.detach(),
        "transition": transition.detach(),
        "gain": gain.detach(),
        "teacher": teacher.detach(),
        "pred_physical": pred_physical.detach(),
    }
