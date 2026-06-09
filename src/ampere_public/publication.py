from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ampere.data.appliance_loader import load_appliance_8ch
from ampere.data.canonical import write_dataframe_prefer_parquet
from ampere.data.rlc_loader import load_rlc_sample


EXPECTED_HASHES = {
    "combined_output.csv": "7a46846804b2ea255b7f2cb041d9f9df2139ccbf5ee851fecdf47ebe038e7cb6",
    "Sample_training_data.csv": "c9fdc042ea2a3438d66d5eb1d94f65c7d2dd39fa2478518f12451985f9fd085e",
}

EXPECTED_RESULTS = {
    "linear_dwell_mae": 75.39441704611784,
    "best_online_tree_dwell_mae": 78.19738926083397,
    "dwellmlp_mean": 43.83111707887881,
    "dwellmlp_std": 2.8978905133964625,
    "dwellobserver_t_mean": 40.64386529001349,
    "dwellobserver_t_std": 1.2933353270642651,
}


class DataVerificationError(RuntimeError):
    """Raised when required public raw data is missing or not reproducible."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_public_root(start: Path | None = None) -> Path:
    root = (start or Path.cwd()).resolve()
    while root != root.parent:
        if (root / "src").is_dir() and (root / "results" / "expected").is_dir():
            return root
        root = root.parent
    raise RuntimeError("Could not locate AMPERE public repo root.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_path(path: str | Path) -> str:
    """Return a repo-relative path string when possible for public outputs."""
    root = resolve_public_root()
    resolved = Path(path)
    if not resolved.is_absolute():
        return resolved.as_posix()
    try:
        return resolved.resolve().relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def sanitize_public_output(text: str) -> str:
    """Remove local checkout and interpreter paths from saved notebook output."""
    root = str(resolve_public_root())
    replacements = {
        root: "<repo root>",
        root.replace("\\", "/"): "<repo root>",
    }
    try:
        env_root = str(Path(sys.executable).resolve().parents[1])
        replacements[env_root] = "<python env>"
        replacements[env_root.replace("\\", "/")] = "<python env>"
    except IndexError:
        pass
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _count_csv_rows(path: Path, *, has_header: bool) -> int:
    with path.open("rb") as handle:
        rows = sum(1 for _ in handle)
    return rows - (1 if has_header else 0)


def verify_data_files(data_dir: str | Path = "data/raw") -> dict[str, dict[str, Any]]:
    root = resolve_public_root()
    base = Path(data_dir)
    if not base.is_absolute():
        base = root / base
    base = base.resolve()

    details: dict[str, dict[str, Any]] = {}
    missing = []
    mismatched = []
    for filename, expected_hash in EXPECTED_HASHES.items():
        path = base / filename
        if not path.is_file():
            missing.append(public_path(path))
            continue
        actual_hash = sha256_file(path)
        if actual_hash.lower() != expected_hash.lower():
            mismatched.append(
                f"{public_path(path)}: expected {expected_hash}, got {actual_hash}"
            )
        details[filename] = {
            "path": public_path(path),
            "sha256": actual_hash,
            "size_bytes": path.stat().st_size,
            "row_count": _count_csv_rows(
                path,
                has_header=filename == "Sample_training_data.csv",
            ),
        }

    if missing or mismatched:
        message = [
            "Required raw data is missing or does not match the published hashes.",
            "Copy the source CSVs into data/raw/ before running verification.",
        ]
        if missing:
            message.append("Missing files:")
            message.extend(f"  - {item}" for item in missing)
        if mismatched:
            message.append("Hash mismatches:")
            message.extend(f"  - {item}" for item in mismatched)
        raise DataVerificationError("\n".join(message))
    return details


def build_public_canonical_outputs(
    data_dir: str | Path = "data/raw",
    output_dir: str | Path = "runs/processed",
) -> dict[str, Any]:
    root = resolve_public_root()
    verify_data_files(data_dir)
    base = Path(data_dir)
    if not base.is_absolute():
        base = root / base
    out = Path(output_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    appliance = load_appliance_8ch(base / "combined_output.csv", root=root)
    rlc = load_rlc_sample(base / "Sample_training_data.csv", root=root)

    outputs = {
        "appliance_8ch_wide": write_dataframe_prefer_parquet(
            appliance.wide, out / "appliance_8ch_wide"
        ),
        "appliance_8ch_long": write_dataframe_prefer_parquet(
            appliance.long, out / "appliance_8ch_long"
        ),
        "rlc_sample_wide": write_dataframe_prefer_parquet(
            rlc.wide, out / "rlc_sample_wide"
        ),
        "rlc_sample_long": write_dataframe_prefer_parquet(
            rlc.long, out / "rlc_sample_long"
        ),
    }
    manifest = {
        "datasets": {
            "appliance_8ch": appliance.metadata,
            "rlc_sample": rlc.metadata,
        },
        "outputs": outputs,
        "expected_hashes": EXPECTED_HASHES,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def run_command(args: list[str], *, cwd: Path | None = None) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(resolve_public_root() / "src")
    printable = []
    for index, arg in enumerate(args):
        if index == 0 and Path(arg).resolve() == Path(sys.executable).resolve():
            printable.append("python")
        else:
            printable.append(public_path(arg))
    print("$ " + " ".join(printable))
    result = subprocess.run(
        args,
        cwd=str(cwd or resolve_public_root()),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout:
        print(sanitize_public_output(result.stdout).rstrip())
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {args}")


def run_corrected_baseline_pipeline(
    data_dir: str | Path = "data/raw",
    output_dir: str | Path = "runs",
) -> dict[str, Path]:
    root = resolve_public_root()
    out = Path(output_dir)
    if not out.is_absolute():
        out = root / out
    processed = out / "processed"
    build_public_canonical_outputs(data_dir=data_dir, output_dir=processed)

    classical_out = out / "reconstruction" / "classical"
    tree_out = out / "reconstruction" / "tree"
    figures = out / "figures"
    reports = out / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            sys.executable,
            "scripts/run_classical_baselines.py",
            "--datasets",
            "appliance_8ch",
            "--processed-dir",
            public_path(processed),
            "--output-dir",
            public_path(classical_out),
            "--figures-dir",
            public_path(figures / "classical"),
            "--report-path",
            public_path(reports / "classical_baseline_report.md"),
        ],
        cwd=root,
    )
    run_command(
        [
            sys.executable,
            "scripts/run_tree_baselines.py",
            "--datasets",
            "appliance_8ch",
            "--modes",
            "online_safe",
            "offline",
            "--target-modes",
            "raw_signed_power",
            "dwell_mean_power",
            "--processed-dir",
            public_path(processed),
            "--classical-dir",
            public_path(classical_out),
            "--output-dir",
            public_path(tree_out),
            "--figures-dir",
            public_path(figures / "tree"),
            "--report-path",
            public_path(reports / "tree_baseline_report.md"),
        ],
        cwd=root,
    )
    return {
        "processed": Path(public_path(processed)),
        "classical": Path(public_path(classical_out)),
        "tree": Path(public_path(tree_out)),
        "reports": Path(public_path(reports)),
    }


def assert_close(name: str, actual: float, expected: float, tolerance: float) -> None:
    delta = abs(float(actual) - float(expected))
    if delta > tolerance:
        raise AssertionError(
            f"{name} mismatch: actual={actual:.8f}, expected={expected:.8f}, "
            f"tolerance={tolerance:.8f}"
        )


def verify_baseline_outputs(
    tree_dir: str | Path = "runs/reconstruction/tree",
    *,
    tolerance_w: float = 0.05,
) -> pd.DataFrame:
    root = resolve_public_root()
    tree_path = Path(tree_dir)
    if not tree_path.is_absolute():
        tree_path = root / tree_path
    comparison = pd.read_csv(tree_path / "tree_vs_classical_comparison.csv")
    row = comparison[comparison["target_mode"].eq("dwell_mean_power")].iloc[0]
    assert_close(
        "linear dwell MAE",
        row["classical_dwell_mae"],
        EXPECTED_RESULTS["linear_dwell_mae"],
        tolerance_w,
    )
    assert_close(
        "best online tree dwell MAE",
        row["online_dwell_mae"],
        EXPECTED_RESULTS["best_online_tree_dwell_mae"],
        tolerance_w,
    )
    return comparison


def verify_expected_claims(expected_dir: str | Path = "results/expected") -> pd.DataFrame:
    root = resolve_public_root()
    expected = Path(expected_dir)
    if not expected.is_absolute():
        expected = root / expected
    headline = pd.read_csv(expected / "corrected_headline_comparison.csv")
    seed = pd.read_csv(expected / "corrected_seed_repeatability.csv")

    linear = headline[headline["family"].eq("Corrected linear")].iloc[0]
    tree = headline[headline["family"].eq("Best online tree")].iloc[0]
    mlp = seed[seed["family"].eq("DwellMLP")].iloc[0]
    observer = seed[seed["family"].eq("DwellObserver-T")].iloc[0]

    assert_close(
        "expected linear dwell MAE",
        linear["dwell_mae"],
        EXPECTED_RESULTS["linear_dwell_mae"],
        1e-9,
    )
    assert_close(
        "expected online tree dwell MAE",
        tree["dwell_mae"],
        EXPECTED_RESULTS["best_online_tree_dwell_mae"],
        1e-9,
    )
    assert_close(
        "expected DwellMLP mean",
        mlp["dwell_mae_mean"],
        EXPECTED_RESULTS["dwellmlp_mean"],
        1e-9,
    )
    assert_close(
        "expected DwellObserver-T mean",
        observer["dwell_mae_mean"],
        EXPECTED_RESULTS["dwellobserver_t_mean"],
        1e-9,
    )
    return headline


def rebuild_summary_plots(
    expected_dir: str | Path = "results/expected",
    output_dir: str | Path = "runs/verified_figures",
) -> list[Path]:
    root = resolve_public_root()
    expected = Path(expected_dir)
    if not expected.is_absolute():
        expected = root / expected
    out = Path(output_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    headline = pd.read_csv(expected / "corrected_headline_comparison.csv")
    seed = pd.read_csv(expected / "corrected_seed_repeatability.csv")
    per_branch = pd.read_csv(expected / "corrected_per_branch_mae_seed_mean.csv")

    paths: list[Path] = []
    labels = ["Linear", "Online tree", "DwellMLP", "DwellObserver-T"]
    values = [
        float(headline[headline["family"].eq("Corrected linear")]["dwell_mae"].iloc[0]),
        float(headline[headline["family"].eq("Best online tree")]["dwell_mae"].iloc[0]),
        float(seed[seed["family"].eq("DwellMLP")]["dwell_mae_mean"].iloc[0]),
        float(seed[seed["family"].eq("DwellObserver-T")]["dwell_mae_mean"].iloc[0]),
    ]
    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"])
    plt.ylabel("Dwell MAE (W)")
    plt.title("Corrected appliance_8ch 40 ms headline comparison")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    path = out / "headline_dwell_mae.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(Path(public_path(path)))

    pivot = per_branch.pivot(index="branch_name", columns="family", values="mae")
    ax = pivot.plot(kind="bar", figsize=(10, 4.8), color=["#54A24B", "#B279A2"])
    ax.set_ylabel("Per-branch MAE (W)")
    ax.set_title("Corrected neural per-branch MAE")
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    path = out / "per_branch_neural_mae.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(Path(public_path(path)))
    return paths
