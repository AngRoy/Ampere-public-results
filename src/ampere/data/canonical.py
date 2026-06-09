from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ampere.utils.paths import as_repo_relative, repo_root


REQUIRED_LONG_COLUMNS = [
    "dataset_id",
    "source_type",
    "time_s",
    "time_index",
    "branch_id",
    "branch_name",
    "load_type",
    "P_raw_signed",
    "P_true",
    "power_representation",
    "P_observed",
    "is_observed",
    "selected_branch",
    "dwell_id",
    "scan_cycle_id",
    "time_since_last_seen",
    "time_to_next_seen",
    "last_observed_value",
    "next_observed_value",
    "dt_s",
    "time_axis_confidence",
    "scaling_applied",
    "source_file",
]

CANONICAL_OPTIONAL_COLUMNS = [
    "dwell_position",
    "P_dwell_mean",
    "P_active_consumption",
    "P_clipped_nonnegative",
    "root_power",
    "dwell_s_requested",
    "dwell_s_effective",
]


@dataclass(frozen=True)
class BranchSpec:
    branch_id: int
    branch_name: str
    load_type: str
    source_column: str


@dataclass(frozen=True)
class Schedule:
    selected_branch: np.ndarray
    dwell_id: np.ndarray
    scan_cycle_id: np.ndarray
    dwell_position: np.ndarray
    dwell_samples: int
    dwell_s_requested: float
    dwell_s_effective: float


def make_round_robin_schedule(
    n_rows: int,
    n_branches: int,
    dwell_samples: int,
    dt_s: float,
    dwell_s_requested: float | None = None,
) -> Schedule:
    if dwell_samples < 1:
        raise ValueError("dwell_samples must be >= 1")
    time_index = np.arange(n_rows, dtype=np.int64)
    dwell_id = time_index // dwell_samples
    selected_branch = (dwell_id % n_branches).astype(np.int64) + 1
    scan_cycle_id = dwell_id // n_branches
    dwell_position = time_index % dwell_samples
    requested = float(dwell_s_requested if dwell_s_requested is not None else dwell_samples * dt_s)
    effective = float(dwell_samples * dt_s)
    return Schedule(
        selected_branch=selected_branch,
        dwell_id=dwell_id.astype(np.int64),
        scan_cycle_id=scan_cycle_id.astype(np.int64),
        dwell_position=dwell_position.astype(np.int64),
        dwell_samples=int(dwell_samples),
        dwell_s_requested=requested,
        dwell_s_effective=effective,
    )


def detect_dt(time_s: pd.Series, sample_size: int = 1000) -> float:
    values = time_s.to_numpy(dtype=float)
    if len(values) < 2:
        raise ValueError("Need at least two timestamps to detect dt")
    diffs = np.diff(values[: min(len(values), sample_size + 1)])
    return float(np.median(diffs))


def observation_features(
    values: np.ndarray,
    is_observed: np.ndarray,
    dt_s: float,
) -> dict[str, np.ndarray]:
    n_rows = len(values)
    last_value = np.full(n_rows, np.nan, dtype=float)
    next_value = np.full(n_rows, np.nan, dtype=float)
    time_since = np.full(n_rows, np.nan, dtype=float)
    time_to_next = np.full(n_rows, np.nan, dtype=float)

    last_idx: int | None = None
    last_val = np.nan
    for idx in range(n_rows):
        if is_observed[idx]:
            last_idx = idx
            last_val = float(values[idx])
        if last_idx is not None:
            last_value[idx] = last_val
            time_since[idx] = (idx - last_idx) * dt_s

    next_idx: int | None = None
    next_val = np.nan
    for idx in range(n_rows - 1, -1, -1):
        if is_observed[idx]:
            next_idx = idx
            next_val = float(values[idx])
        if next_idx is not None:
            next_value[idx] = next_val
            time_to_next[idx] = (next_idx - idx) * dt_s

    return {
        "last_observed_value": last_value,
        "next_observed_value": next_value,
        "time_since_last_seen": time_since,
        "time_to_next_seen": time_to_next,
    }


def dwell_mean(values: np.ndarray, dwell_id: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"dwell_id": dwell_id, "value": values})
    return frame.groupby("dwell_id")["value"].transform("mean").to_numpy(dtype=float)


def canonical_long_from_wide(
    *,
    wide: pd.DataFrame,
    branch_specs: list[BranchSpec],
    dataset_id: str,
    source_type: str,
    dt_s: float,
    source_file: str,
    time_axis_confidence: str,
    scaling_applied: str,
    power_representation: str = "raw_signed_power",
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    selected_branch = wide["selected_branch"].to_numpy(dtype=np.int64)
    dwell_id = wide["dwell_id"].to_numpy(dtype=np.int64)

    for spec in branch_specs:
        values = wide[f"P_raw_signed_b{spec.branch_id:02d}"].to_numpy(dtype=float)
        observed_mask = selected_branch == spec.branch_id
        features = observation_features(values, observed_mask, dt_s)
        observed = np.where(observed_mask, wide["P_observed"].to_numpy(dtype=float), np.nan)
        branch_df = pd.DataFrame(
            {
                "dataset_id": dataset_id,
                "source_type": source_type,
                "time_s": wide["time_s"].to_numpy(dtype=float),
                "time_index": wide["time_index"].to_numpy(dtype=np.int64),
                "branch_id": spec.branch_id,
                "branch_name": spec.branch_name,
                "load_type": spec.load_type,
                "P_raw_signed": values,
                "P_true": values,
                "power_representation": power_representation,
                "P_observed": observed,
                "is_observed": observed_mask,
                "selected_branch": selected_branch,
                "dwell_id": dwell_id,
                "scan_cycle_id": wide["scan_cycle_id"].to_numpy(dtype=np.int64),
                "dwell_position": wide["dwell_position"].to_numpy(dtype=np.int64),
                "time_since_last_seen": features["time_since_last_seen"],
                "time_to_next_seen": features["time_to_next_seen"],
                "last_observed_value": features["last_observed_value"],
                "next_observed_value": features["next_observed_value"],
                "dt_s": dt_s,
                "time_axis_confidence": time_axis_confidence,
                "scaling_applied": scaling_applied,
                "source_file": source_file,
                "P_dwell_mean": dwell_mean(values, dwell_id),
                "P_active_consumption": np.nan,
                "P_clipped_nonnegative": np.maximum(values, 0.0),
                "root_power": wide["root_power"].to_numpy(dtype=float),
                "dwell_s_requested": wide["dwell_s_requested"].to_numpy(dtype=float),
                "dwell_s_effective": wide["dwell_s_effective"].to_numpy(dtype=float),
            }
        )
        parts.append(branch_df)

    long = pd.concat(parts, ignore_index=True)
    return long[REQUIRED_LONG_COLUMNS + CANONICAL_OPTIONAL_COLUMNS]


def write_dataframe_prefer_parquet(df: pd.DataFrame, output_stem: Path) -> dict[str, Any]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = output_stem.with_suffix(".parquet")
    csv_path = output_stem.with_suffix(".csv")
    try:
        df.to_parquet(parquet_path, index=False)
        return {
            "path": str(parquet_path),
            "format": "parquet",
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
        }
    except Exception as exc:
        df.to_csv(csv_path, index=False)
        return {
            "path": str(csv_path),
            "format": "csv",
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "parquet_error": repr(exc),
        }


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or repo_root() / "configs" / "dataset_config.yaml"
    if not path.exists():
        return {}
    try:
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except ImportError:
        return {}


def build_phase1_outputs(
    *,
    root: Path | None = None,
    config_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    from ampere.data.appliance_loader import load_appliance_8ch
    from ampere.data.rlc_loader import load_rlc_sample

    base = root or repo_root()
    config = load_config(config_path)
    # On this Windows workspace, `data/` collides with the existing raw `Data/`
    # directory. Keep generated artifacts out of the raw-data tree.
    configured_output = config.get("processed_output_path", "outputs/data/processed")
    outputs = output_dir or base / configured_output

    rlc_cfg = config.get("rlc", {})
    appliance_cfg = config.get("appliance_8ch", {})

    rlc_path = base / rlc_cfg.get(
        "sample_path",
        "data/raw/Sample_training_data.csv",
    )
    appliance_path = base / appliance_cfg.get(
        "combined_output_path",
        "data/raw/combined_output.csv",
    )

    rlc = load_rlc_sample(rlc_path, root=base)
    appliance = load_appliance_8ch(appliance_path, root=base)

    written = {
        "rlc_sample_wide": write_dataframe_prefer_parquet(rlc.wide, outputs / "rlc_sample_wide"),
        "rlc_sample_long": write_dataframe_prefer_parquet(rlc.long, outputs / "rlc_sample_long"),
        "appliance_8ch_wide": write_dataframe_prefer_parquet(
            appliance.wide,
            outputs / "appliance_8ch_wide",
        ),
        "appliance_8ch_long": write_dataframe_prefer_parquet(
            appliance.long,
            outputs / "appliance_8ch_long",
        ),
    }

    manifest = {
        "phase": "phase1_data_validation_and_canonicalization",
        "generated_by": "scripts/build_canonical_dataset.py",
        "datasets": {
            "rlc_sample": rlc.metadata,
            "appliance_8ch": appliance.metadata,
        },
        "outputs": written,
        "policy": {
            "rlc_default_power_representation": "raw_signed_power",
            "clipping_policy": "P_clipped_nonnegative is diagnostic only; P_true is not clipped",
            "appliance_time_axis": "dt_s=0.04 from time-major interleaved dense truth: 45000 common rows over 30 minutes",
        },
    }
    outputs.mkdir(parents=True, exist_ok=True)
    manifest_path = outputs / "manifest.json"
    manifest["manifest_path"] = as_repo_relative(manifest_path, base)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
