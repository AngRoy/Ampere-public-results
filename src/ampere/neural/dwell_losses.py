from __future__ import annotations

from dataclasses import dataclass

from ampere.neural.dwell_dataset import DwellSeries
from ampere.neural.utils import require_torch

torch = require_torch()
import torch.nn.functional as F


DWELL_LOSS_VARIANTS = (
    "dwell_supervised",
    "dwell_energy",
    "dwell_observed_consistency",
    "dwell_nonnegative_valid",
    "dwell_full",
)


@dataclass(frozen=True)
class DwellLossWeights:
    supervised: float = 1.0
    energy: float = 0.0
    observed: float = 0.0
    nonnegative: float = 0.0


def dwell_nonnegativity_allowed(series: DwellSeries) -> bool:
    return series.dataset_id == "appliance_8ch" and series.target_mode in {"dwell_mean_power", "raw_signed_power"}


def dwell_loss_weights(variant: str, series: DwellSeries) -> DwellLossWeights:
    if variant not in DWELL_LOSS_VARIANTS:
        raise ValueError(f"Unknown DwellNet loss variant: {variant}")
    if variant == "dwell_supervised":
        return DwellLossWeights()
    if variant == "dwell_energy":
        return DwellLossWeights(energy=0.02)
    if variant == "dwell_observed_consistency":
        return DwellLossWeights(observed=0.10)
    if variant == "dwell_nonnegative_valid":
        return DwellLossWeights(nonnegative=0.01 if dwell_nonnegativity_allowed(series) else 0.0)
    return DwellLossWeights(
        energy=0.02,
        observed=0.10,
        nonnegative=0.01 if dwell_nonnegativity_allowed(series) else 0.0,
    )


def decode_dwell_prediction(prediction, batch: dict[str, object], series: DwellSeries):
    if series.output_mode == "absolute":
        return series.output_normalizer.inverse_torch(prediction)
    residual = series.output_normalizer.inverse_torch(prediction)
    return batch["base_physical"] + residual


def dwell_weighted_energy_loss(pred_physical, target_physical, dwell_s_effective, epsilon: float = 1e-6):
    dwell_s = dwell_s_effective.view(-1, 1)
    e_pred = pred_physical * dwell_s / 3600.0
    e_true = target_physical * dwell_s / 3600.0
    numerator = torch.sum(torch.abs(e_pred - e_true), dim=1)
    denominator = epsilon + torch.sum(torch.abs(e_true), dim=1)
    return torch.mean(numerator / denominator)


def observed_dwell_consistency_loss(pred_physical, observed_values, observed_mask):
    mask = observed_mask > 0.5
    if not bool(mask.any()):
        return pred_physical.sum() * 0.0
    return F.smooth_l1_loss(pred_physical[mask], observed_values[mask])


def dwell_nonnegative_loss(pred_physical, *, enabled: bool):
    if not enabled:
        return pred_physical.sum() * 0.0
    return torch.relu(-pred_physical).mean()


def dwellnet_loss(
    *,
    prediction,
    batch: dict[str, object],
    series: DwellSeries,
    variant: str,
) -> dict[str, object]:
    weights = dwell_loss_weights(variant, series)
    pred_physical = decode_dwell_prediction(prediction, batch, series)
    supervised = F.smooth_l1_loss(prediction, batch["target"])
    energy = dwell_weighted_energy_loss(pred_physical, batch["target_physical"], batch["dwell_s_effective"])
    observed = observed_dwell_consistency_loss(pred_physical, batch["observed_values"], batch["observed_mask"])
    nonnegative = dwell_nonnegative_loss(pred_physical, enabled=weights.nonnegative > 0.0)
    total = (
        weights.supervised * supervised
        + weights.energy * energy
        + weights.observed * observed
        + weights.nonnegative * nonnegative
    )
    return {
        "total": total,
        "supervised": supervised.detach(),
        "energy": energy.detach(),
        "observed": observed.detach(),
        "nonnegative": nonnegative.detach(),
        "pred_physical": pred_physical.detach(),
    }
