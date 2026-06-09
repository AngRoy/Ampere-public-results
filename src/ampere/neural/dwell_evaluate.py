from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ampere.evaluation.leaderboard import rank_leaderboard
from ampere.evaluation.metrics import MetricTables, evaluate_predictions
from ampere.neural.dwell_dataset import DwellSeries


DWELLNET_PREDICTION_COLUMNS = [
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
    "model_family",
    "neural_architecture",
    "model_level",
    "reconstruction_mode",
    "target_mode",
    "loss_variant",
    "P_hat",
    "is_observed",
    "P_observed",
    "selected_branch",
    "dt_s",
    "split",
    "source_file",
    "window_cycles",
    "output_mode",
    "normalization_mode",
    "use_branch_embeddings",
    "use_time_features",
]


def dwellnet_method_name(metadata: dict[str, Any]) -> str:
    suffix = []
    if not bool(metadata.get("use_branch_embeddings", True)):
        suffix.append("noemb")
    if not bool(metadata.get("include_time_features", True)):
        suffix.append("notime")
    core = (
        f"{metadata['model_type']}_{metadata['loss_variant']}_{metadata['output_mode']}_"
        f"wc{metadata['window_cycles']}_{metadata['normalization_mode']}"
    )
    return core if not suffix else core + "_" + "_".join(suffix)


def build_dwell_prediction_frame(
    series: DwellSeries,
    dwell_predictions: np.ndarray,
    *,
    metadata: dict[str, Any],
    split: str = "test",
) -> pd.DataFrame:
    frame = series.long[series.long["split"].astype(str).eq(split)].copy()
    branch_to_pos = {branch_id: idx for idx, branch_id in enumerate(series.branch_ids)}
    dwell_to_pos = {int(dwell_id): idx for idx, dwell_id in enumerate(series.dwell_id)}
    rows = frame[["dwell_id", "branch_id"]].to_numpy(dtype=np.int64)
    p_hat = []
    p_true = []
    for dwell_id, branch_id in rows:
        dwell_pos = dwell_to_pos[int(dwell_id)]
        branch_pos = branch_to_pos[int(branch_id)]
        value = dwell_predictions[dwell_pos, branch_pos]
        if not np.isfinite(value):
            value = series.last_observed_physical[dwell_pos, branch_pos]
        if not np.isfinite(value):
            value = series.target_normalizer.mean[branch_pos]
        p_hat.append(float(value))
        p_true.append(float(series.target_physical[dwell_pos, branch_pos]))

    frame["P_true"] = p_true
    frame["P_hat"] = p_hat
    frame["power_representation"] = series.target_mode
    frame["method"] = dwellnet_method_name(metadata)
    frame["model_family"] = "neural"
    frame["neural_architecture"] = str(metadata["model_type"])
    frame["model_level"] = "dwell_sequence"
    frame["reconstruction_mode"] = series.reconstruction_mode
    frame["target_mode"] = series.target_mode
    frame["loss_variant"] = str(metadata["loss_variant"])
    frame["window_cycles"] = int(metadata["window_cycles"])
    frame["output_mode"] = str(metadata["output_mode"])
    frame["normalization_mode"] = str(metadata["normalization_mode"])
    frame["use_branch_embeddings"] = bool(metadata["use_branch_embeddings"])
    frame["use_time_features"] = bool(metadata["include_time_features"])

    optional = [
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
    return frame[DWELLNET_PREDICTION_COLUMNS + [column for column in optional if column in frame.columns]]


def validate_dwell_prediction_schema(predictions: pd.DataFrame) -> None:
    missing = [column for column in DWELLNET_PREDICTION_COLUMNS if column not in predictions.columns]
    if missing:
        raise AssertionError(f"DwellNet predictions missing required columns: {missing}")
    key = [
        "dataset_id",
        "method",
        "target_mode",
        "reconstruction_mode",
        "time_index",
        "branch_id",
    ]
    duplicates = int(predictions.duplicated(key).sum())
    if duplicates:
        raise AssertionError(f"DwellNet predictions contain duplicate key rows: {duplicates}")
    if predictions["P_hat"].isna().any():
        raise AssertionError("DwellNet predictions contain null P_hat")


def _transition_thresholds(series: DwellSeries) -> np.ndarray:
    train = series.target_physical[series.split == "train"]
    diffs = np.abs(np.diff(train, axis=0))
    thresholds = np.nanpercentile(diffs, 75, axis=0) if len(diffs) else np.ones(series.branch_count)
    thresholds = np.where(np.isfinite(thresholds), thresholds, 0.0)
    return thresholds.astype(float)


def transition_stable_metrics(predictions: pd.DataFrame, series: DwellSeries) -> pd.DataFrame:
    thresholds = _transition_thresholds(series)
    rows = []
    group_cols = ["dataset_id", "method", "reconstruction_mode", "target_mode", "neural_architecture", "loss_variant"]
    branch_to_pos = {branch_id: idx for idx, branch_id in enumerate(series.branch_ids)}
    dwell_true = {
        (int(dwell_id), int(branch_id)): float(series.target_physical[pos, branch_to_pos[int(branch_id)]])
        for pos, dwell_id in enumerate(series.dwell_id)
        for branch_id in series.branch_ids
    }
    transition_mask = []
    for row in predictions.itertuples(index=False):
        branch_pos = branch_to_pos[int(row.branch_id)]
        prev = dwell_true.get((int(row.dwell_id) - 1, int(row.branch_id)), getattr(row, "P_true"))
        change = abs(float(getattr(row, "P_true")) - float(prev))
        transition_mask.append(change >= thresholds[branch_pos])
    tmp = predictions.copy()
    tmp["is_transition_region"] = transition_mask
    tmp["abs_error"] = np.abs(tmp["P_hat"].to_numpy(dtype=float) - tmp["P_true"].to_numpy(dtype=float))
    for keys, group in tmp.groupby(group_cols, sort=False):
        key = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        transition = group[group["is_transition_region"]]
        stable = group[~group["is_transition_region"]]
        row = {
            **key,
            "transition_mae": float(transition["abs_error"].mean()) if not transition.empty else float("nan"),
            "stable_mae": float(stable["abs_error"].mean()) if not stable.empty else float("nan"),
            "transition_rows": int(len(transition)),
            "stable_rows": int(len(stable)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_dwellnet_predictions(predictions: pd.DataFrame, series: DwellSeries) -> tuple[MetricTables, pd.DataFrame]:
    validate_dwell_prediction_schema(predictions)
    metrics = evaluate_predictions(predictions)
    transitions = transition_stable_metrics(predictions, series)
    return metrics, transitions


def summarize_ablation(leaderboard: pd.DataFrame) -> pd.DataFrame:
    if leaderboard.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["dataset_id", "target_mode", "reconstruction_mode"]
    for keys, group in leaderboard.groupby(group_cols, sort=False):
        best = group.sort_values("mae").iloc[0]
        for column in ["neural_architecture", "loss_variant", "output_mode", "normalization_mode", "window_cycles"]:
            if column not in group.columns:
                continue
            summary = (
                group.groupby(column, sort=False)
                .agg(best_mae=("mae", "min"), mean_mae=("mae", "mean"), runs=("method", "count"))
                .reset_index()
            )
            for row in summary.itertuples(index=False):
                rows.append(
                    {
                        "dataset_id": keys[0],
                        "target_mode": keys[1],
                        "reconstruction_mode": keys[2],
                        "ablation_axis": column,
                        "value": getattr(row, column),
                        "best_mae": row.best_mae,
                        "mean_mae": row.mean_mae,
                        "runs": int(row.runs),
                        "delta_best_vs_overall_best": row.best_mae - best["mae"],
                    }
                )
    return pd.DataFrame(rows)


def load_matching_baselines(
    *,
    dataset_id: str,
    target_mode: str,
    reconstruction_mode: str,
    tree_dir: Path,
    constraint_dir: Path,
    classical_dir: Path,
) -> pd.DataFrame:
    rows = []
    tree_path = tree_dir / "tree_leaderboard.csv"
    if tree_path.exists():
        tree = pd.read_csv(tree_path)
        tree = tree[
            tree["dataset_id"].eq(dataset_id)
            & tree["target_mode"].eq(target_mode)
            & tree["reconstruction_mode"].eq(reconstruction_mode)
        ].sort_values("mae")
        if not tree.empty:
            row = tree.iloc[0]
            rows.append(
                {
                    "baseline_family": "tree",
                    "method": row["method"],
                    "mae": row["mae"],
                    "dwell_mae": row["dwell_mae"],
                    "weighted_energy_error": row["weighted_energy_error"],
                }
            )
    constraint_path = constraint_dir / "constraint_leaderboard.csv"
    if constraint_path.exists():
        constraint = pd.read_csv(constraint_path)
        constraint = constraint[
            constraint["dataset_id"].eq(dataset_id)
            & constraint["target_mode"].eq(target_mode)
            & constraint["reconstruction_mode"].isin([reconstruction_mode, "classical"])
        ].sort_values("mae")
        if not constraint.empty:
            row = constraint.iloc[0]
            rows.append(
                {
                    "baseline_family": "constraint",
                    "method": row["method"],
                    "mae": row["mae"],
                    "dwell_mae": row["dwell_mae"],
                    "weighted_energy_error": row["weighted_energy_error"],
                }
            )
    classical_path = classical_dir / "classical_leaderboard.csv"
    if classical_path.exists():
        classical = pd.read_csv(classical_path)
        classical = classical[classical["dataset_id"].eq(dataset_id)].sort_values("mae")
        if not classical.empty:
            row = classical.iloc[0]
            rows.append(
                {
                    "baseline_family": "classical",
                    "method": row["method"],
                    "mae": row["mae"],
                    "dwell_mae": row["dwell_mae"],
                    "weighted_energy_error": row["weighted_energy_error"],
                }
            )
    return pd.DataFrame(rows)


def compare_dwellnet_to_baselines(leaderboard: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if leaderboard.empty:
        return pd.DataFrame()
    if baselines.empty or "baseline_family" not in baselines.columns:
        best_tree = pd.DataFrame()
    else:
        best_tree = baselines[baselines["baseline_family"].eq("tree")].head(1)
    tree_mae = float(best_tree["mae"].iloc[0]) if not best_tree.empty else float("nan")
    tree_method = str(best_tree["method"].iloc[0]) if not best_tree.empty else ""
    for row in leaderboard.itertuples(index=False):
        rows.append(
            {
                "dataset_id": row.dataset_id,
                "target_mode": row.target_mode,
                "reconstruction_mode": row.reconstruction_mode,
                "method": row.method,
                "dwellnet_mae": row.mae,
                "dwellnet_dwell_mae": row.dwell_mae,
                "dwellnet_weighted_energy_error": row.weighted_energy_error,
                "best_tree_method": tree_method,
                "tree_mae": tree_mae,
                "beats_tree_mae": bool(np.isfinite(tree_mae) and row.mae < tree_mae),
                "mae_delta_vs_tree": row.mae - tree_mae if np.isfinite(tree_mae) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def rank_dwellnet(metrics: MetricTables) -> pd.DataFrame:
    return rank_leaderboard(metrics.overall)
