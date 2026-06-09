from __future__ import annotations

from dataclasses import dataclass

from ampere.reconstruction.interpolation import pchip_status


@dataclass(frozen=True)
class BaselineSpec:
    method: str
    label: str
    description: str
    edge_policy: str


BASELINE_SPECS = {
    "mean_per_branch": BaselineSpec(
        method="mean_per_branch",
        label="Mean per branch",
        description="Predict each branch's mean observed value, with observed timestamps forced exact.",
        edge_policy="No edge interpolation; all missing timestamps use the branch observed mean.",
    ),
    "zoh": BaselineSpec(
        method="zoh",
        label="ZOH",
        description="Last observation hold per branch.",
        edge_policy="Before the first observation, use that branch's first observed value.",
    ),
    "linear": BaselineSpec(
        method="linear",
        label="Linear interpolation",
        description="Piecewise linear interpolation between observed points.",
        edge_policy="Before/after observed support, use nearest boundary value.",
    ),
    "pchip": BaselineSpec(
        method="pchip",
        label="PCHIP",
        description="Shape-preserving cubic Hermite interpolation from scipy when available.",
        edge_policy="Before/after observed support, use nearest boundary value.",
    ),
    "slope_aware_an": BaselineSpec(
        method="slope_aware_an",
        label="Slope-aware AN",
        description="Practical bounded local-slope Hermite baseline using neighboring observations.",
        edge_policy="Before/after observed support, use nearest boundary value.",
    ),
}

DEFAULT_METHODS = ["mean_per_branch", "zoh", "linear", "pchip", "slope_aware_an"]


def available_methods(methods: list[str] | None = None) -> tuple[list[str], dict[str, str]]:
    requested = list(methods or DEFAULT_METHODS)
    unavailable: dict[str, str] = {}
    if "pchip" in requested:
        status = pchip_status()
        if not status.available:
            requested = [method for method in requested if method != "pchip"]
            unavailable["pchip"] = status.reason
    unknown = [method for method in requested if method not in BASELINE_SPECS]
    if unknown:
        raise ValueError(f"Unknown baseline methods: {unknown}")
    return requested, unavailable
