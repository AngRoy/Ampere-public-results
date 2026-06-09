from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ampere.data.canonical import (
    BranchSpec,
    canonical_long_from_wide,
    make_round_robin_schedule,
)
from ampere.utils.paths import as_repo_relative, repo_root


APPLIANCE_BRANCH_SPECS = [
    BranchSpec(1, "Branch01_Ceiling_Fan", "component_circuit", "Branch01_Ceiling_Fan"),
    BranchSpec(2, "Branch02_Tubelight", "component_circuit", "Branch02_Tubelight"),
    BranchSpec(3, "Branch03_Electric_Kettle", "component_circuit", "Branch03_Electric_Kettle"),
    BranchSpec(4, "Branch04_Electric_Geyser", "component_circuit", "Branch04_Electric_Geyser"),
    BranchSpec(5, "Branch05_Water_Pump", "component_circuit", "Branch05_Water_Pump"),
    BranchSpec(6, "Branch06_Refrigerator", "component_circuit", "Branch06_Refrigerator"),
    BranchSpec(7, "Branch07_Rice_Cooker", "component_circuit", "Branch07_Rice_Cooker"),
    BranchSpec(8, "Branch08_Microwave_Oven", "component_circuit", "Branch08_Microwave_Oven"),
]

APPLIANCE_BRANCH_COUNT = 8
APPLIANCE_SAMPLES_PER_BRANCH = 45_000
APPLIANCE_DURATION_S = 30 * 60
APPLIANCE_DT_S = APPLIANCE_DURATION_S / APPLIANCE_SAMPLES_PER_BRANCH
APPLIANCE_POWER_SCALE = 501_530.0
APPLIANCE_NATIVE_DWELL_S = APPLIANCE_DT_S


@dataclass
class CanonicalDataset:
    wide: pd.DataFrame
    long: pd.DataFrame
    metadata: dict


def read_interleaved_values(path: str | Path) -> np.ndarray:
    df = pd.read_csv(path, header=None, names=["value"])
    values = pd.to_numeric(df["value"], errors="raise").to_numpy(dtype=float)
    if len(values) % APPLIANCE_BRANCH_COUNT != 0:
        raise ValueError(
            f"Interleaved appliance value count {len(values)} is not divisible by "
            f"{APPLIANCE_BRANCH_COUNT}"
        )
    return values


def appliance_wide_from_values(
    values: np.ndarray,
    *,
    source_file: str,
    root: str | Path | None = None,
    dwell_s_requested: float = APPLIANCE_NATIVE_DWELL_S,
) -> tuple[pd.DataFrame, dict]:
    base = Path(root).resolve() if root is not None else repo_root()
    reshaped = values.reshape(-1, APPLIANCE_BRANCH_COUNT)
    n_rows = reshaped.shape[0]
    dt_s = APPLIANCE_DT_S
    dwell_samples = max(1, int(round(dwell_s_requested / dt_s)))
    schedule = make_round_robin_schedule(
        n_rows,
        n_branches=APPLIANCE_BRANCH_COUNT,
        dwell_samples=dwell_samples,
        dt_s=dt_s,
        dwell_s_requested=dwell_s_requested,
    )
    time_index = np.arange(n_rows, dtype=np.int64)
    time_s = time_index * dt_s
    scaled = reshaped * APPLIANCE_POWER_SCALE

    selected_zero = schedule.selected_branch - 1
    observed = scaled[np.arange(n_rows), selected_zero]
    wide = pd.DataFrame(
        {
            "dataset_id": "appliance_8ch",
            "source_type": "component_circuit",
            "time_s": time_s,
            "time_index": time_index,
            "selected_branch": schedule.selected_branch,
            "P_observed": observed,
            "dwell_id": schedule.dwell_id,
            "scan_cycle_id": schedule.scan_cycle_id,
            "dwell_position": schedule.dwell_position,
            "dt_s": dt_s,
            "dwell_s_requested": schedule.dwell_s_requested,
            "dwell_s_effective": schedule.dwell_s_effective,
            "time_axis_confidence": "time_major_interleaved_dense_truth_40ms",
            "power_representation": "raw_signed_power",
            "scaling_applied": f"power_x{APPLIANCE_POWER_SCALE:g}",
            "source_file": as_repo_relative(source_file, base),
        }
    )
    root_power = np.zeros(n_rows, dtype=float)
    for idx, spec in enumerate(APPLIANCE_BRANCH_SPECS):
        raw_unscaled = reshaped[:, idx]
        raw_scaled = scaled[:, idx]
        wide[f"P_unscaled_b{spec.branch_id:02d}"] = raw_unscaled
        wide[f"P_raw_signed_b{spec.branch_id:02d}"] = raw_scaled
        wide[f"P_true_b{spec.branch_id:02d}"] = raw_scaled
        wide[f"P_clipped_nonnegative_b{spec.branch_id:02d}"] = np.maximum(raw_scaled, 0.0)
        root_power += raw_scaled
    wide["root_power"] = root_power

    metadata = {
        "dataset_id": "appliance_8ch",
        "source_type": "component_circuit",
        "source_file": as_repo_relative(source_file, base),
        "raw_value_count": int(len(values)),
        "rows": int(n_rows),
        "wide_shape": [int(wide.shape[0]), int(wide.shape[1])],
        "branch_count": APPLIANCE_BRANCH_COUNT,
        "branch_names": [spec.branch_name for spec in APPLIANCE_BRANCH_SPECS],
        "samples_per_branch": int(n_rows),
        "expected_samples_per_branch": APPLIANCE_SAMPLES_PER_BRANCH,
        "duration_s": APPLIANCE_DURATION_S,
        "dt_s": dt_s,
        "time_axis_confidence": "time_major_interleaved_dense_truth_40ms",
        "power_scale": APPLIANCE_POWER_SCALE,
        "dwell_s_requested": schedule.dwell_s_requested,
        "dwell_samples": schedule.dwell_samples,
        "dwell_s_effective": schedule.dwell_s_effective,
        "scan_cycle_s": schedule.dwell_s_effective * APPLIANCE_BRANCH_COUNT,
        "dwell_policy": "native 40 ms physical scan mask: one selected branch per 0.04 s row",
        "power_representation": "raw_signed_power",
    }
    return wide, metadata


def load_appliance_8ch(
    path: str | Path | None = None,
    *,
    root: str | Path | None = None,
    dwell_s_requested: float = APPLIANCE_NATIVE_DWELL_S,
) -> CanonicalDataset:
    base = Path(root).resolve() if root is not None else repo_root()
    source_path = (
        Path(path)
        if path is not None
        else base / "data/raw/combined_output.csv"
    )
    values = read_interleaved_values(source_path)
    wide, metadata = appliance_wide_from_values(
        values,
        source_file=str(source_path),
        root=base,
        dwell_s_requested=dwell_s_requested,
    )
    long = canonical_long_from_wide(
        wide=wide,
        branch_specs=APPLIANCE_BRANCH_SPECS,
        dataset_id="appliance_8ch",
        source_type="component_circuit",
        dt_s=metadata["dt_s"],
        source_file=metadata["source_file"],
        time_axis_confidence="time_major_interleaved_dense_truth_40ms",
        scaling_applied=f"power_x{APPLIANCE_POWER_SCALE:g}",
        power_representation="raw_signed_power",
    )
    return CanonicalDataset(wide=wide, long=long, metadata=metadata)
