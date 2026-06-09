from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from ampere.data.canonical import REQUIRED_LONG_COLUMNS, write_dataframe_prefer_parquet
from ampere.reconstruction.baselines import BASELINE_SPECS, DEFAULT_METHODS, available_methods
from ampere.reconstruction.interpolation import (
    linear_reconstruct,
    mean_per_branch_reconstruct,
    pchip_reconstruct,
    slope_aware_an_reconstruct,
    zoh_reconstruct,
)


PREDICTION_REQUIRED_COLUMNS = [
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
    "P_hat",
    "is_observed",
    "P_observed",
    "selected_branch",
    "dt_s",
    "split",
    "source_file",
]

PREDICTION_OPTIONAL_COLUMNS = [
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

METHOD_FUNCTIONS: dict[str, Callable] = {
    "mean_per_branch": mean_per_branch_reconstruct,
    "zoh": zoh_reconstruct,
    "linear": linear_reconstruct,
    "pchip": pchip_reconstruct,
    "slope_aware_an": slope_aware_an_reconstruct,
}


@dataclass(frozen=True)
class ClassicalRunResult:
    predictions: pd.DataFrame
    method_metadata: dict[str, object]


def validate_canonical_input(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_LONG_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Canonical input missing required columns: {missing}")
    if "raw_signed_power" not in set(df["power_representation"].dropna().unique()):
        raise ValueError("Classical baselines currently expect raw_signed_power canonical inputs")
    if df.duplicated(subset=["dataset_id", "time_index", "branch_id"]).any():
        raise ValueError("Canonical input has duplicate dataset_id/time_index/branch_id rows")


def load_canonical_long(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    elif input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported canonical input format: {input_path}")
    validate_canonical_input(df)
    return df


def prediction_columns_for(df: pd.DataFrame) -> list[str]:
    columns = list(PREDICTION_REQUIRED_COLUMNS)
    for column in PREDICTION_OPTIONAL_COLUMNS:
        if column in df.columns and column not in columns:
            columns.append(column)
    return columns


def reconstruct_branch(group: pd.DataFrame, method: str) -> pd.DataFrame:
    func = METHOD_FUNCTIONS[method]
    ordered = group.sort_values("time_index").copy()
    x = ordered["time_index"].to_numpy()
    observed_mask = ordered["is_observed"].to_numpy(dtype=bool)
    observed_values = ordered["P_observed"].to_numpy(dtype=float)
    ordered["P_hat"] = func(x, observed_mask, observed_values)
    ordered["method"] = method
    if "split" not in ordered.columns:
        ordered["split"] = pd.NA
    return ordered


def run_classical_methods(
    canonical_long: pd.DataFrame,
    methods: list[str] | None = None,
) -> ClassicalRunResult:
    validate_canonical_input(canonical_long)
    selected_methods, unavailable = available_methods(methods or DEFAULT_METHODS)
    metadata: dict[str, object] = {
        "requested_methods": methods or DEFAULT_METHODS,
        "executed_methods": selected_methods,
        "unavailable_methods": unavailable,
        "method_descriptions": {
            method: {
                "label": BASELINE_SPECS[method].label,
                "description": BASELINE_SPECS[method].description,
                "edge_policy": BASELINE_SPECS[method].edge_policy,
            }
            for method in selected_methods
        },
    }
    if not selected_methods:
        columns = prediction_columns_for(canonical_long)
        return ClassicalRunResult(predictions=pd.DataFrame(columns=columns), method_metadata=metadata)

    parts: list[pd.DataFrame] = []
    for method in selected_methods:
        branch_parts = [
            reconstruct_branch(group, method)
            for _, group in canonical_long.groupby(["dataset_id", "branch_id"], sort=False)
        ]
        method_df = pd.concat(branch_parts, ignore_index=True)
        parts.append(method_df[prediction_columns_for(method_df)])

    predictions = pd.concat(parts, ignore_index=True)
    return ClassicalRunResult(predictions=predictions, method_metadata=metadata)


def write_predictions(predictions: pd.DataFrame, output_stem: str | Path) -> dict[str, object]:
    return write_dataframe_prefer_parquet(predictions, Path(output_stem))
