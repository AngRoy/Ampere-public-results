from __future__ import annotations

from pathlib import Path

import pandas as pd


def _safe_name(value: str) -> str:
    return (
        value.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
        .lower()
    )


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt, None
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, f"matplotlib unavailable: {exc!r}"


def plot_reconstruction_overlay(
    predictions: pd.DataFrame,
    *,
    dataset_id: str,
    branch_id: int,
    output_dir: Path,
    seconds: float = 30.0,
) -> Path | None:
    plt, error = _load_matplotlib()
    if error:
        return None

    branch = predictions[
        (predictions["dataset_id"] == dataset_id) & (predictions["branch_id"] == branch_id)
    ].copy()
    if branch.empty:
        return None
    start = float(branch["time_s"].min())
    window = branch[branch["time_s"] <= start + seconds]
    if window.empty:
        return None

    base = window[window["method"] == window["method"].iloc[0]]
    branch_name = str(base["branch_name"].iloc[0])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(base["time_s"], base["P_true"], color="black", linewidth=1.4, label="P_true")
    observed = base[base["is_observed"].astype(bool)]
    ax.scatter(
        observed["time_s"],
        observed["P_observed"],
        color="#d62728",
        s=14,
        label="observed sparse points",
        zorder=5,
    )
    for method, method_df in window.groupby("method", sort=False):
        ax.plot(method_df["time_s"], method_df["P_hat"], linewidth=1.0, alpha=0.85, label=method)
    ax.set_title(f"{dataset_id} branch {branch_id}: {branch_name}")
    ax.set_xlabel("time_s")
    ax.set_ylabel("power, raw_signed_power")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{_safe_name(dataset_id)}_branch{branch_id:02d}_overlay.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_method_mae_bar(leaderboard: pd.DataFrame, *, dataset_id: str, output_dir: Path) -> Path | None:
    plt, error = _load_matplotlib()
    if error:
        return None
    data = leaderboard[leaderboard["dataset_id"] == dataset_id].sort_values("mae")
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(data["method"], data["mae"], color="#4c78a8")
    ax.set_title(f"{dataset_id}: method MAE")
    ax.set_xlabel("method")
    ax.set_ylabel("MAE")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{_safe_name(dataset_id)}_method_mae.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_branch_energy_error(
    branch_metrics: pd.DataFrame,
    *,
    dataset_id: str,
    method: str,
    output_dir: Path,
) -> Path | None:
    plt, error = _load_matplotlib()
    if error:
        return None
    data = branch_metrics[
        (branch_metrics["dataset_id"] == dataset_id) & (branch_metrics["method"] == method)
    ].sort_values("branch_id")
    if data.empty:
        return None
    labels = [f"{int(row.branch_id)}" for row in data.itertuples(index=False)]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(labels, data["abs_energy_error_Wh"], color="#f58518")
    ax.set_title(f"{dataset_id}: per-branch energy error ({method})")
    ax.set_xlabel("branch_id")
    ax.set_ylabel("absolute signed-energy error, Wh")
    ax.grid(axis="y", alpha=0.25)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{_safe_name(dataset_id)}_{_safe_name(method)}_branch_energy_error.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_classical_plots(
    *,
    predictions: pd.DataFrame,
    leaderboard: pd.DataFrame,
    branch_metrics: pd.DataFrame,
    output_dir: str | Path,
) -> list[str]:
    out = Path(output_dir)
    generated: list[str] = []
    for dataset_id, dataset_predictions in predictions.groupby("dataset_id", sort=False):
        branch_ids = sorted(dataset_predictions["branch_id"].unique().tolist())[:2]
        for branch_id in branch_ids:
            path = plot_reconstruction_overlay(
                predictions,
                dataset_id=str(dataset_id),
                branch_id=int(branch_id),
                output_dir=out,
            )
            if path is not None:
                generated.append(str(path))
        path = plot_method_mae_bar(leaderboard, dataset_id=str(dataset_id), output_dir=out)
        if path is not None:
            generated.append(str(path))
        best = leaderboard[leaderboard["dataset_id"] == dataset_id].sort_values("rank").head(1)
        if not best.empty:
            method = str(best["method"].iloc[0])
            path = plot_branch_energy_error(
                branch_metrics,
                dataset_id=str(dataset_id),
                method=method,
                output_dir=out,
            )
            if path is not None:
                generated.append(str(path))
    return generated
