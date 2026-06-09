from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from ampere.data.canonical import (
    BranchSpec,
    canonical_long_from_wide,
    detect_dt,
    make_round_robin_schedule,
)
from ampere.utils.paths import as_repo_relative, repo_root


RLC_BRANCH_SPECS = [
    BranchSpec(1, "Washing machine", "RL", "Washing machine"),
    BranchSpec(2, "Electric Iron", "R", "Electric Iron"),
    BranchSpec(3, "LED Tv", "RC", "LED Tv"),
    BranchSpec(4, "CFL Lamp", "RL", "CFL Lamp"),
]

RLC_COLUMNS = ["Time", *[spec.source_column for spec in RLC_BRANCH_SPECS], "MCP_data"]
RLC_DT_S = 0.04
RLC_DWELL_SAMPLES = 25


@dataclass
class CanonicalDataset:
    wide: pd.DataFrame
    long: pd.DataFrame
    metadata: dict


def read_rlc_csv(path: str | Path, *, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=nrows)
    missing = [col for col in RLC_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"RLC CSV missing expected columns: {missing}")
    df = df[RLC_COLUMNS].copy()
    for col in RLC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df


def iter_rlc_chunks(path: str | Path, *, chunksize: int) -> Iterator[pd.DataFrame]:
    if chunksize < 1:
        raise ValueError("chunksize must be >= 1")
    for chunk in pd.read_csv(path, chunksize=chunksize):
        missing = [col for col in RLC_COLUMNS if col not in chunk.columns]
        if missing:
            raise ValueError(f"RLC CSV missing expected columns: {missing}")
        chunk = chunk[RLC_COLUMNS].copy()
        for col in RLC_COLUMNS:
            chunk[col] = pd.to_numeric(chunk[col], errors="raise")
        yield chunk


def iter_rlc_full_chunks(
    path: str | Path | None = None,
    *,
    root: str | Path | None = None,
    chunksize: int = 100_000,
) -> Iterator[pd.DataFrame]:
    base = Path(root).resolve() if root is not None else repo_root()
    source_path = (
        Path(path)
        if path is not None
        else base / "Data/Simple_RLC_Circuits/power_consumption_merged_training_data.csv"
    )
    yield from iter_rlc_chunks(source_path, chunksize=chunksize)


def selected_truth_values(df: pd.DataFrame, selected_branch: np.ndarray) -> np.ndarray:
    values = np.empty(len(df), dtype=float)
    for spec in RLC_BRANCH_SPECS:
        mask = selected_branch == spec.branch_id
        values[mask] = df.loc[mask, spec.source_column].to_numpy(dtype=float)
    return values


def infer_selected_branch_from_mcp(df: pd.DataFrame, *, atol: float = 1e-9) -> np.ndarray:
    matches = np.zeros((len(df), len(RLC_BRANCH_SPECS)), dtype=bool)
    mcp = df["MCP_data"].to_numpy(dtype=float)
    for idx, spec in enumerate(RLC_BRANCH_SPECS):
        matches[:, idx] = np.isclose(
            df[spec.source_column].to_numpy(dtype=float),
            mcp,
            atol=atol,
            rtol=0.0,
        )
    inferred = np.full(len(df), -1, dtype=np.int64)
    unique = matches.sum(axis=1) == 1
    inferred[unique] = matches[unique].argmax(axis=1) + 1
    return inferred


def rlc_wide_from_frame(
    df: pd.DataFrame,
    *,
    dataset_id: str,
    source_file: str,
    root: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    base = Path(root).resolve() if root is not None else repo_root()
    dt_s = detect_dt(df["Time"])
    schedule = make_round_robin_schedule(
        len(df),
        n_branches=len(RLC_BRANCH_SPECS),
        dwell_samples=RLC_DWELL_SAMPLES,
        dt_s=dt_s,
        dwell_s_requested=1.0,
    )
    selected_values = selected_truth_values(df, schedule.selected_branch)
    mcp = df["MCP_data"].to_numpy(dtype=float)
    alignment_error = np.abs(selected_values - mcp)

    wide = pd.DataFrame(
        {
            "dataset_id": dataset_id,
            "source_type": "rlc_sim",
            "time_s": df["Time"].to_numpy(dtype=float),
            "time_index": np.arange(len(df), dtype=np.int64),
            "selected_branch": schedule.selected_branch,
            "P_observed": mcp,
            "dwell_id": schedule.dwell_id,
            "scan_cycle_id": schedule.scan_cycle_id,
            "dwell_position": schedule.dwell_position,
            "dt_s": dt_s,
            "dwell_s_requested": schedule.dwell_s_requested,
            "dwell_s_effective": schedule.dwell_s_effective,
            "time_axis_confidence": "detected_from_time_column",
            "power_representation": "raw_signed_power",
            "scaling_applied": "none",
            "source_file": as_repo_relative(source_file, base),
        }
    )
    root_power = np.zeros(len(df), dtype=float)
    for spec in RLC_BRANCH_SPECS:
        raw = df[spec.source_column].to_numpy(dtype=float)
        wide[f"P_raw_signed_b{spec.branch_id:02d}"] = raw
        wide[f"P_true_b{spec.branch_id:02d}"] = raw
        wide[f"P_clipped_nonnegative_b{spec.branch_id:02d}"] = np.maximum(raw, 0.0)
        root_power += raw
    wide["root_power"] = root_power

    inferred = infer_selected_branch_from_mcp(df)
    known_pattern = schedule.selected_branch
    metadata = {
        "dataset_id": dataset_id,
        "source_type": "rlc_sim",
        "source_file": as_repo_relative(source_file, base),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "branch_count": len(RLC_BRANCH_SPECS),
        "branch_names": [spec.branch_name for spec in RLC_BRANCH_SPECS],
        "dt_s": dt_s,
        "expected_dt_s": RLC_DT_S,
        "dt_error_s": abs(dt_s - RLC_DT_S),
        "samples_per_second": round(1.0 / dt_s),
        "dwell_samples": RLC_DWELL_SAMPLES,
        "dwell_s_requested": 1.0,
        "dwell_s_effective": schedule.dwell_s_effective,
        "scan_cycle_s": schedule.dwell_s_effective * len(RLC_BRANCH_SPECS),
        "mcp_alignment_max_abs_error": float(alignment_error.max()),
        "mcp_alignment_error_count_gt_1e_9": int(np.sum(alignment_error > 1e-9)),
        "mcp_inference_unique_fraction": float(np.mean(inferred > 0)),
        "mcp_inference_matches_known_schedule_fraction": float(
            np.mean((inferred == known_pattern) | (inferred < 0))
        ),
        "time_axis_confidence": "detected_from_time_column",
        "power_representation": "raw_signed_power",
    }
    return wide, metadata


def load_rlc_sample(
    path: str | Path | None = None,
    *,
    root: str | Path | None = None,
    nrows: int | None = None,
) -> CanonicalDataset:
    base = Path(root).resolve() if root is not None else repo_root()
    source_path = Path(path) if path is not None else base / "data/raw/Sample_training_data.csv"
    df = read_rlc_csv(source_path, nrows=nrows)
    wide, metadata = rlc_wide_from_frame(
        df,
        dataset_id="rlc_sample",
        source_file=str(source_path),
        root=base,
    )
    long = canonical_long_from_wide(
        wide=wide,
        branch_specs=RLC_BRANCH_SPECS,
        dataset_id="rlc_sample",
        source_type="rlc_sim",
        dt_s=metadata["dt_s"],
        source_file=metadata["source_file"],
        time_axis_confidence="detected_from_time_column",
        scaling_applied="none",
        power_representation="raw_signed_power",
    )
    return CanonicalDataset(wide=wide, long=long, metadata=metadata)


def load_rlc_full_sample(
    path: str | Path | None = None,
    *,
    root: str | Path | None = None,
    nrows: int = 5000,
) -> CanonicalDataset:
    base = Path(root).resolve() if root is not None else repo_root()
    source_path = (
        Path(path)
        if path is not None
        else base / "Data/Simple_RLC_Circuits/power_consumption_merged_training_data.csv"
    )
    df = read_rlc_csv(source_path, nrows=nrows)
    wide, metadata = rlc_wide_from_frame(
        df,
        dataset_id="rlc_full_sampled",
        source_file=str(source_path),
        root=base,
    )
    metadata["load_mode"] = "bounded_sample"
    metadata["requested_rows"] = int(nrows)
    long = canonical_long_from_wide(
        wide=wide,
        branch_specs=RLC_BRANCH_SPECS,
        dataset_id="rlc_full_sampled",
        source_type="rlc_sim",
        dt_s=metadata["dt_s"],
        source_file=metadata["source_file"],
        time_axis_confidence="detected_from_time_column",
        scaling_applied="none",
        power_representation="raw_signed_power",
    )
    return CanonicalDataset(wide=wide, long=long, metadata=metadata)


def load_rlc_full_canonical(
    path: str | Path | None = None,
    *,
    root: str | Path | None = None,
    allow_full: bool = False,
) -> CanonicalDataset:
    """Canonicalize the full RLC corpus only when explicitly opted in."""

    if not allow_full:
        raise ValueError(
            "Full RLC canonicalization is disabled by default. "
            "Use load_rlc_full_sample or iter_rlc_full_chunks, or pass allow_full=True."
        )
    base = Path(root).resolve() if root is not None else repo_root()
    source_path = (
        Path(path)
        if path is not None
        else base / "Data/Simple_RLC_Circuits/power_consumption_merged_training_data.csv"
    )
    df = read_rlc_csv(source_path)
    wide, metadata = rlc_wide_from_frame(
        df,
        dataset_id="rlc_full",
        source_file=str(source_path),
        root=base,
    )
    metadata["load_mode"] = "explicit_full"
    long = canonical_long_from_wide(
        wide=wide,
        branch_specs=RLC_BRANCH_SPECS,
        dataset_id="rlc_full",
        source_type="rlc_sim",
        dt_s=metadata["dt_s"],
        source_file=metadata["source_file"],
        time_axis_confidence="detected_from_time_column",
        scaling_applied="none",
        power_representation="raw_signed_power",
    )
    return CanonicalDataset(wide=wide, long=long, metadata=metadata)
