from __future__ import annotations

from dataclasses import dataclass

from ampere.neural.utils import PowerScaler, require_torch

torch = require_torch()
import torch.nn.functional as F


LOSS_VARIANTS = (
    "neural_base",
    "neural_obs",
    "neural_obs_energy",
    "neural_obs_energy_dwell",
    "neural_full_valid",
)


@dataclass(frozen=True)
class LossWeights:
    supervised: float = 1.0
    observed: float = 0.0
    energy: float = 0.0
    dwell: float = 0.0
    nonnegative: float = 0.0
    calibration: float = 0.0


def nonnegativity_allowed(dataset_id: str, target_mode: str) -> bool:
    return dataset_id == "appliance_8ch" and target_mode in {"raw_signed_power", "dwell_mean_power"}


def weights_for_variant(*, variant: str, dataset_id: str, target_mode: str) -> LossWeights:
    if variant not in LOSS_VARIANTS:
        raise ValueError(f"Unknown neural loss variant: {variant}")
    dwell_weight = 0.05 if target_mode == "dwell_mean_power" else 0.0
    if variant == "neural_base":
        return LossWeights()
    if variant == "neural_obs":
        return LossWeights(observed=0.20)
    if variant == "neural_obs_energy":
        return LossWeights(observed=0.20, energy=0.05)
    if variant == "neural_obs_energy_dwell":
        return LossWeights(observed=0.20, energy=0.05, dwell=dwell_weight)
    return LossWeights(
        observed=0.20,
        energy=0.05,
        dwell=dwell_weight,
        nonnegative=0.02 if nonnegativity_allowed(dataset_id, target_mode) else 0.0,
        calibration=1e-4,
    )


def observed_consistency_loss(prediction, observed_values, observed_mask):
    mask = observed_mask > 0.5
    if not bool(mask.any()):
        return prediction.sum() * 0.0
    return F.smooth_l1_loss(prediction[mask], observed_values[mask])


def weighted_energy_loss(prediction, target, dt_s, scaler: PowerScaler, epsilon: float = 1e-6):
    pred_phys = scaler.inverse_torch(prediction)
    target_phys = scaler.inverse_torch(target)
    dt = dt_s.unsqueeze(1)
    e_pred = torch.sum(pred_phys * dt, dim=2) / 3600.0
    e_true = torch.sum(target_phys * dt, dim=2) / 3600.0
    numerator = torch.sum(torch.abs(e_pred - e_true), dim=1)
    denominator = epsilon + torch.sum(torch.abs(e_true), dim=1)
    return torch.mean(numerator / denominator)


def dwell_consistency_loss(prediction, dwell_id):
    penalties = []
    for batch_idx in range(prediction.shape[0]):
        ids = torch.unique(dwell_id[batch_idx])
        for dwell in ids:
            mask = dwell_id[batch_idx] == dwell
            if int(mask.sum().item()) < 2:
                continue
            segment = prediction[batch_idx, :, mask]
            penalties.append(torch.mean((segment - segment.mean(dim=1, keepdim=True)) ** 2))
    if not penalties:
        return prediction.sum() * 0.0
    return torch.stack(penalties).mean()


def nonnegative_loss(prediction, scaler: PowerScaler, *, enabled: bool):
    if not enabled:
        return prediction.sum() * 0.0
    pred_phys = scaler.inverse_torch(prediction)
    return torch.relu(-pred_phys).mean()


def physics_guided_loss(
    *,
    prediction,
    target,
    observed_values,
    observed_mask,
    dt_s,
    dwell_id,
    scaler: PowerScaler,
    weights: LossWeights,
    calibration_regularization=None,
) -> dict[str, object]:
    supervised = F.smooth_l1_loss(prediction, target)
    observed = observed_consistency_loss(prediction, observed_values, observed_mask)
    energy = weighted_energy_loss(prediction, target, dt_s, scaler)
    dwell = dwell_consistency_loss(prediction, dwell_id)
    nonnegative = nonnegative_loss(prediction, scaler, enabled=weights.nonnegative > 0.0)
    calibration = (
        calibration_regularization()
        if calibration_regularization is not None
        else torch.as_tensor(0.0, dtype=prediction.dtype, device=prediction.device)
    )
    total = (
        weights.supervised * supervised
        + weights.observed * observed
        + weights.energy * energy
        + weights.dwell * dwell
        + weights.nonnegative * nonnegative
        + weights.calibration * calibration
    )
    return {
        "total": total,
        "supervised": supervised.detach(),
        "observed": observed.detach(),
        "energy": energy.detach(),
        "dwell": dwell.detach(),
        "nonnegative": nonnegative.detach(),
        "calibration": calibration.detach(),
    }
