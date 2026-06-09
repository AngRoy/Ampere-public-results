from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ampere.data.appliance_loader import (
    APPLIANCE_BRANCH_COUNT,
    APPLIANCE_DT_S,
    APPLIANCE_POWER_SCALE,
    APPLIANCE_SAMPLES_PER_BRANCH,
    APPLIANCE_BRANCH_SPECS,
)
from ampere.data.canonical import REQUIRED_LONG_COLUMNS
from ampere.data.rlc_loader import RLC_BRANCH_SPECS, RLC_DT_S, RLC_DWELL_SAMPLES
from ampere.utils.paths import as_repo_relative, repo_root


def assert_close(actual: float, expected: float, *, tolerance: float, label: str) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def validate_rlc_dataset(dataset: Any) -> dict[str, Any]:
    meta = dataset.metadata
    wide = dataset.wide
    long = dataset.long

    assert meta["branch_count"] == len(RLC_BRANCH_SPECS)
    assert_close(meta["dt_s"], RLC_DT_S, tolerance=1e-9, label="RLC dt")
    assert meta["samples_per_second"] == 25
    assert meta["dwell_samples"] == RLC_DWELL_SAMPLES
    assert meta["mcp_alignment_error_count_gt_1e_9"] == 0
    expected_selected = ((wide["time_index"] // RLC_DWELL_SAMPLES) % len(RLC_BRANCH_SPECS)) + 1
    if not np.array_equal(wide["selected_branch"].to_numpy(), expected_selected.to_numpy()):
        raise AssertionError("RLC selected branch pattern mismatch")
    validate_canonical_long(long)
    return {
        "rows_wide": int(len(wide)),
        "rows_long": int(len(long)),
        "dt_s": meta["dt_s"],
        "dwell_samples": meta["dwell_samples"],
        "mcp_alignment_max_abs_error": meta["mcp_alignment_max_abs_error"],
        "branch_count": meta["branch_count"],
    }


def validate_appliance_dataset(dataset: Any) -> dict[str, Any]:
    meta = dataset.metadata
    wide = dataset.wide
    long = dataset.long

    assert meta["raw_value_count"] == 360_000
    assert meta["rows"] == APPLIANCE_SAMPLES_PER_BRANCH
    assert meta["branch_count"] == APPLIANCE_BRANCH_COUNT
    assert_close(meta["dt_s"], APPLIANCE_DT_S, tolerance=1e-12, label="appliance dt")
    assert meta["dwell_samples"] == 1
    assert_close(meta["dwell_s_effective"], 0.04, tolerance=1e-12, label="appliance dwell")
    assert_close(meta["scan_cycle_s"], 0.32, tolerance=1e-12, label="appliance scan cycle")
    expected_names = [spec.branch_name for spec in APPLIANCE_BRANCH_SPECS]
    assert meta["branch_names"] == expected_names
    first_scaled = wide.loc[0, "P_raw_signed_b01"]
    first_unscaled = wide.loc[0, "P_unscaled_b01"]
    assert_close(
        first_scaled,
        first_unscaled * APPLIANCE_POWER_SCALE,
        tolerance=1e-9,
        label="appliance scaling",
    )
    validate_canonical_long(long)
    return {
        "raw_value_count": meta["raw_value_count"],
        "wide_shape": list(wide.shape),
        "long_shape": list(long.shape),
        "dt_s": meta["dt_s"],
        "dwell_samples": meta["dwell_samples"],
        "dwell_s_effective": meta["dwell_s_effective"],
        "scan_cycle_s": meta["scan_cycle_s"],
        "branch_count": meta["branch_count"],
    }


def validate_canonical_long(df: pd.DataFrame) -> dict[str, Any]:
    missing = [col for col in REQUIRED_LONG_COLUMNS if col not in df.columns]
    if missing:
        raise AssertionError(f"Canonical long missing columns: {missing}")
    duplicated = df.duplicated(subset=["dataset_id", "time_index", "branch_id"]).sum()
    if duplicated:
        raise AssertionError(f"Canonical long has {duplicated} duplicate key rows")
    if df["power_representation"].isna().any():
        raise AssertionError("power_representation contains nulls")
    if not (df["power_representation"] == "raw_signed_power").all():
        raise AssertionError("Phase 1 outputs must use raw_signed_power as P_true")
    if not np.allclose(df["P_true"], df["P_raw_signed"], equal_nan=True):
        raise AssertionError("P_true must equal P_raw_signed for raw_signed_power outputs")
    observed = df["is_observed"].astype(bool)
    if df.loc[observed, "P_observed"].isna().any():
        raise AssertionError("Observed rows have null P_observed")
    if df.loc[~observed, "P_observed"].notna().any():
        raise AssertionError("Unobserved rows must have null P_observed")
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "required_columns": REQUIRED_LONG_COLUMNS,
        "power_representations": sorted(df["power_representation"].unique().tolist()),
    }


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    stat = p.stat()
    return {
        "path": str(p),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def assert_fingerprint_unchanged(before: dict[str, Any], after: dict[str, Any]) -> None:
    if before["size"] != after["size"] or before["mtime_ns"] != after["mtime_ns"]:
        raise AssertionError(f"Raw file changed: {before['path']}")


def write_phase1_report(
    *,
    path: str | Path,
    manifest: dict[str, Any],
    validation: dict[str, Any],
    root: str | Path | None = None,
) -> None:
    base = Path(root).resolve() if root is not None else repo_root()
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    outputs = manifest["outputs"]
    lines = [
        "# Phase 1 Data Validation Report",
        "",
        "Generated by `python scripts/validate_data.py`.",
        "",
        "## Scope",
        "",
        "Phase 1 created reusable data loaders, canonical long/wide outputs, and validation checks. No ML models, baselines, RL policies, or physics-guided models were trained or implemented.",
        "",
        "## Loaded Inputs",
        "",
        f"- RLC sample: `{manifest['datasets']['rlc_sample']['source_file']}`",
        f"- Component-circuit 8-channel dataset: `{manifest['datasets']['appliance_8ch']['source_file']}`",
        "- Full RLC month corpus: supported by bounded/chunked loader, not processed by default.",
        "",
        "## Generated Outputs",
        "",
        "The requested `data/processed` path collides with raw `Data/` on this Windows filesystem, so generated files are written under `outputs/data/processed` to keep raw data untouched.",
        "",
    ]
    for name, item in outputs.items():
        lines.append(f"- `{name}`: `{as_repo_relative(item['path'], base)}` ({item['format']}, {item['rows']} rows x {item['columns']} columns)")
    lines.append(f"- manifest: `{manifest['manifest_path']}`")
    lines.extend(
        [
            "",
            "## Dataset Shapes",
            "",
            f"- RLC sample wide: {validation['rlc_sample']['rows_wide']} rows",
            f"- RLC sample long: {validation['rlc_sample']['rows_long']} rows",
            f"- Appliance wide: {validation['appliance_8ch']['wide_shape'][0]} rows x {validation['appliance_8ch']['wide_shape'][1]} columns",
            f"- Appliance long: {validation['appliance_8ch']['long_shape'][0]} rows x {validation['appliance_8ch']['long_shape'][1]} columns",
            "",
            "## dt And Dwell Validation",
            "",
            f"- RLC dt: {validation['rlc_sample']['dt_s']} s, 25 samples/s, dwell = {validation['rlc_sample']['dwell_samples']} samples = 1.0 s.",
            f"- RLC MCP alignment max absolute error: {validation['rlc_sample']['mcp_alignment_max_abs_error']}.",
            "- RLC selected branch follows the 25-sample round-robin schedule.",
            f"- Appliance dt: {validation['appliance_8ch']['dt_s']} s from 45,000 common time rows over 30 minutes.",
            f"- Appliance native scan mask: dwell_samples = {validation['appliance_8ch']['dwell_samples']}, effective dwell = {validation['appliance_8ch']['dwell_s_effective']} s, 8-branch scan cycle = {validation['appliance_8ch']['scan_cycle_s']} s.",
            "",
            "## RLC Negative-Power Policy",
            "",
            "`P_raw_signed` preserves raw RLC source values. `P_true` equals `P_raw_signed` because `power_representation = raw_signed_power`. `P_clipped_nonnegative` is generated only as a diagnostic/ablation column and is not the default target. `P_active_consumption` remains undefined.",
            "",
            "## Appliance Time-Axis Policy",
            "",
            "`combined_output.csv` is interpreted as time-major interleaved dense truth: rows 1-8 are appliances 1-8 for the same 40 ms interval, rows 9-16 for the next interval, and so on. The canonical time axis uses `dt_s = 0.04` and `time_axis_confidence = time_major_interleaved_dense_truth_40ms`.",
            "",
            "## Schema Summary",
            "",
            "Long-form outputs include explicit dataset/source metadata, branch metadata, `P_raw_signed`, `P_true`, `power_representation`, observation mask fields, dwell/scan ids, last/next observation features, dt, scaling, source file, dwell metadata, and diagnostic clipped power.",
            "",
            "## Known Limitations",
            "",
            "- Full 30-day RLC corpus is not canonicalized by default.",
            "- Appliance 8-channel observation mask is generated as a native 40 ms physical scan mask over dense branch truth.",
            "- Appliance row timestamps are reconstructed from the documented time-major interleaved layout.",
            "- Generated processed outputs use `outputs/data/processed` instead of `data/processed` because `data` and `Data` are the same path on this Windows workspace.",
            "- Active consumption power needs a separate confirmed derivation.",
            "",
            "## Recommended Next Phase",
            "",
            "Phase 2 can implement classical reconstruction baselines against these canonical outputs: ZOH, linear interpolation, PCHIP, and a slope/change-aware baseline. Keep `raw_signed_power` and `dwell_mean_power` evaluations separate.",
            "",
            "## Validation JSON",
            "",
            "```json",
            json.dumps(validation, indent=2),
            "```",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
