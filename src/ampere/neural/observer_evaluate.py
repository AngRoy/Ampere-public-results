from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ampere.evaluation.leaderboard import rank_leaderboard
from ampere.evaluation.metrics import MetricTables, evaluate_predictions
from ampere.neural.dwell_evaluate import DWELLNET_PREDICTION_COLUMNS, transition_stable_metrics
from ampere.neural.observer_dataset import ObserverSeries


OBSERVER_PREDICTION_COLUMNS = DWELLNET_PREDICTION_COLUMNS + ["hard_observed_projection"]
OBSERVER_EVAL_MODES = ("student", "ema_teacher", "student_teacher_average")


def observer_method_name(metadata: dict[str, Any]) -> str:
    suffix = []
    if not bool(metadata.get("use_branch_embeddings", True)):
        suffix.append("noemb")
    if not bool(metadata.get("include_time_features", True)):
        suffix.append("notime")
    if not bool(metadata.get("hard_observed_projection", True)):
        suffix.append("unprojected")
    transition_weight = float(metadata.get("transition_weight", 0.0) or 0.0)
    gain_lambda = float(metadata.get("gain_lambda", 0.0) or 0.0)
    if metadata.get("loss_variant") == "observer_transition":
        suffix.append(f"tw{transition_weight:g}")
    if gain_lambda > 0.0:
        suffix.append(f"gl{gain_lambda:g}")
    if bool(metadata.get("use_ema_target", False)):
        suffix.append(f"ema_tau{float(metadata.get('ema_tau', 0.0)):g}")
        suffix.append(f"tl{float(metadata.get('teacher_lambda', 0.0)):g}")
        teacher_mask = str(metadata.get("teacher_mask", "all"))
        teacher_space = str(metadata.get("teacher_space", "absolute"))
        suffix.append(f"tm{teacher_mask}")
        suffix.append(f"ts{teacher_space}")
        warmup = int(metadata.get("teacher_warmup_epochs", 0) or 0)
        ramp = int(metadata.get("teacher_ramp_epochs", 0) or 0)
        if warmup > 0:
            suffix.append(f"twarm{warmup}")
        if ramp > 0:
            suffix.append(f"tramp{ramp}")
        perturbation = str(metadata.get("teacher_perturbation", "none"))
        if perturbation != "none":
            suffix.append(f"tp{perturbation}")
    eval_mode = str(metadata.get("observer_eval_mode", "student"))
    if eval_mode != "student":
        suffix.append(f"eval_{eval_mode}")
    if bool(metadata.get("method_seed_suffix", False)):
        suffix.append(f"seed{int(metadata.get('seed', 0))}")
    core = (
        f"{metadata['model_type']}_{metadata['loss_variant']}_{metadata['output_mode']}_"
        f"wc{metadata['window_cycles']}_{metadata['normalization_mode']}"
    )
    return core if not suffix else core + "_" + "_".join(suffix)


def build_observer_prediction_frame(
    series: ObserverSeries,
    dwell_predictions: np.ndarray,
    *,
    metadata: dict[str, Any],
    split: str = "test",
) -> pd.DataFrame:
    frame = series.dwell.long[series.dwell.long["split"].astype(str).eq(split)].copy()
    branch_to_pos = {branch_id: idx for idx, branch_id in enumerate(series.dwell.branch_ids)}
    dwell_to_pos = {int(dwell_id): idx for idx, dwell_id in enumerate(series.dwell.dwell_id)}
    p_hat = []
    p_true = []
    for dwell_id, branch_id in frame[["dwell_id", "branch_id"]].to_numpy(dtype=np.int64):
        dwell_pos = dwell_to_pos[int(dwell_id)]
        branch_pos = branch_to_pos[int(branch_id)]
        value = dwell_predictions[dwell_pos, branch_pos]
        if not np.isfinite(value):
            value = series.dwell.last_observed_physical[dwell_pos, branch_pos]
        if not np.isfinite(value):
            value = series.dwell.target_normalizer.mean[branch_pos]
        p_hat.append(float(value))
        p_true.append(float(series.dwell.target_physical[dwell_pos, branch_pos]))

    frame["P_true"] = p_true
    frame["P_hat"] = p_hat
    frame["power_representation"] = series.target_mode
    frame["method"] = observer_method_name(metadata)
    config_metadata = dict(metadata)
    config_metadata["observer_eval_mode"] = "student"
    config_metadata["method_seed_suffix"] = False
    frame["observer_config_method"] = observer_method_name(config_metadata)
    frame["model_family"] = "neural"
    frame["neural_architecture"] = str(metadata["model_type"])
    frame["model_level"] = "dwell_observer"
    frame["reconstruction_mode"] = series.reconstruction_mode
    frame["target_mode"] = series.target_mode
    frame["loss_variant"] = str(metadata["loss_variant"])
    frame["window_cycles"] = int(metadata["window_cycles"])
    frame["output_mode"] = str(metadata["output_mode"])
    frame["normalization_mode"] = str(metadata["normalization_mode"])
    frame["transition_weight"] = float(metadata.get("transition_weight", 0.0) or 0.0)
    frame["gain_lambda"] = float(metadata.get("gain_lambda", 0.0) or 0.0)
    frame["use_ema_target"] = bool(metadata.get("use_ema_target", False))
    frame["ema_tau"] = float(metadata.get("ema_tau", 0.0) or 0.0)
    frame["teacher_lambda"] = float(metadata.get("teacher_lambda", 0.0) or 0.0)
    frame["teacher_mask"] = str(metadata.get("teacher_mask", "none"))
    frame["teacher_space"] = str(metadata.get("teacher_space", "none"))
    frame["teacher_warmup_epochs"] = int(metadata.get("teacher_warmup_epochs", 0) or 0)
    frame["teacher_ramp_epochs"] = int(metadata.get("teacher_ramp_epochs", 0) or 0)
    frame["teacher_perturbation"] = str(metadata.get("teacher_perturbation", "none"))
    frame["observer_eval_mode"] = str(metadata.get("observer_eval_mode", "student"))
    frame["seed"] = int(metadata.get("seed", 0) or 0)
    frame["use_branch_embeddings"] = bool(metadata["use_branch_embeddings"])
    frame["use_time_features"] = bool(metadata["include_time_features"])
    frame["hard_observed_projection"] = bool(metadata.get("hard_observed_projection", True))

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
        "transition_weight",
        "gain_lambda",
        "use_ema_target",
        "ema_tau",
        "teacher_lambda",
        "teacher_mask",
        "teacher_space",
        "teacher_warmup_epochs",
        "teacher_ramp_epochs",
        "teacher_perturbation",
        "observer_eval_mode",
        "observer_config_method",
        "seed",
    ]
    return frame[OBSERVER_PREDICTION_COLUMNS + [column for column in optional if column in frame.columns]]


def validate_observer_prediction_schema(predictions: pd.DataFrame) -> None:
    missing = [column for column in OBSERVER_PREDICTION_COLUMNS if column not in predictions.columns]
    if missing:
        raise AssertionError(f"DwellObserver predictions missing required columns: {missing}")
    key = ["dataset_id", "method", "target_mode", "reconstruction_mode", "time_index", "branch_id"]
    if "seed" in predictions.columns:
        key.append("seed")
    duplicates = int(predictions.duplicated(key).sum())
    if duplicates:
        raise AssertionError(f"DwellObserver predictions contain duplicate key rows: {duplicates}")
    if predictions["P_hat"].isna().any():
        raise AssertionError("DwellObserver predictions contain null P_hat")


def build_gain_frame(
    series: ObserverSeries,
    gains: np.ndarray,
    innovations: np.ndarray,
    *,
    metadata: dict[str, Any],
    split: str = "test",
) -> pd.DataFrame:
    method = observer_method_name(metadata)
    rows = []
    branch_to_pos = {branch_id: idx for idx, branch_id in enumerate(series.dwell.branch_ids)}
    split_positions = np.flatnonzero(series.dwell.split == split)
    for pos in split_positions:
        if not np.isfinite(innovations[pos]):
            continue
        selected_branch = int(series.dwell.selected_branch[pos])
        selected_pos = branch_to_pos[selected_branch]
        for branch_id in series.dwell.branch_ids:
            branch_pos = branch_to_pos[int(branch_id)]
            gain = gains[pos, branch_pos]
            if not np.isfinite(gain):
                continue
            rows.append(
                {
                    "dataset_id": series.dataset_id,
                    "target_mode": series.target_mode,
                    "reconstruction_mode": series.reconstruction_mode,
                    "method": method,
                    "neural_architecture": metadata["model_type"],
                    "loss_variant": metadata["loss_variant"],
                    "dwell_id": int(series.dwell.dwell_id[pos]),
                    "selected_branch": selected_branch,
                    "branch_id": int(branch_id),
                    "branch_name": series.dwell.branch_names[branch_pos],
                    "gain": float(gain),
                    "abs_gain": float(abs(gain)),
                    "is_selected_branch": bool(branch_pos == selected_pos),
                    "innovation_scaled": float(innovations[pos]),
                }
            )
    return pd.DataFrame(rows)


def gain_diagnostics(gain_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if gain_frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    for keys, group in gain_frame.groupby(["dataset_id", "target_mode", "reconstruction_mode", "method"], sort=False):
        selected = group[group["is_selected_branch"]]
        off = group[~group["is_selected_branch"]]
        rows.append(
            {
                "dataset_id": keys[0],
                "target_mode": keys[1],
                "reconstruction_mode": keys[2],
                "method": keys[3],
                "mean_selected_branch_gain": float(selected["gain"].mean()) if not selected.empty else float("nan"),
                "mean_off_branch_abs_gain": float(off["abs_gain"].mean()) if not off.empty else float("nan"),
                "max_abs_gain": float(group["abs_gain"].max()),
                "mean_abs_innovation_scaled": float(group.drop_duplicates("dwell_id")["innovation_scaled"].abs().mean()),
            }
        )
    by_selected = (
        gain_frame.groupby(["method", "selected_branch", "branch_id", "branch_name", "is_selected_branch"], sort=False)
        .agg(mean_gain=("gain", "mean"), mean_abs_gain=("abs_gain", "mean"), rows=("gain", "count"))
        .reset_index()
    )
    return pd.DataFrame(rows), by_selected


def evaluate_observer_predictions(predictions: pd.DataFrame, series: ObserverSeries) -> tuple[MetricTables, pd.DataFrame]:
    validate_observer_prediction_schema(predictions)
    metrics = evaluate_predictions(predictions)
    transitions = transition_stable_metrics(predictions, series.dwell)
    return metrics, transitions


def rank_observer(metrics: MetricTables | pd.DataFrame) -> pd.DataFrame:
    overall = metrics.overall if isinstance(metrics, MetricTables) else metrics
    return rank_leaderboard(overall)


def reference_baselines() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "baseline_family": "residual_prior_zoh",
                "method": "residual_prior_only",
                "dwell_mae": 72.7865,
                "unobserved_mae": 83.1845,
                "weighted_energy_error": 0.0056,
                "transition_mae": np.nan,
                "stable_mae": np.nan,
            },
            {
                "baseline_family": "classical",
                "method": "linear",
                "dwell_mae": 67.4753,
                "unobserved_mae": 77.1147,
                "weighted_energy_error": 0.0032,
                "transition_mae": np.nan,
                "stable_mae": np.nan,
            },
            {
                "baseline_family": "tree",
                "method": "window_dwell_extra_trees",
                "dwell_mae": 64.1049,
                "unobserved_mae": 73.2590,
                "weighted_energy_error": 0.0093,
                "transition_mae": np.nan,
                "stable_mae": np.nan,
            },
            {
                "baseline_family": "constraint",
                "method": "window_dwell_extra_trees__dwell_consistent",
                "dwell_mae": 64.1049,
                "unobserved_mae": 73.2590,
                "weighted_energy_error": 0.0093,
                "transition_mae": np.nan,
                "stable_mae": np.nan,
            },
            {
                "baseline_family": "stage3c_r1_no_ema",
                "method": "observer_mlp_prior_only_observer_supervised_residual_dwell_wc8_branchwise",
                "dwell_mae": 34.8116,
                "unobserved_mae": 39.7847,
                "weighted_energy_error": 0.0091,
                "transition_mae": 71.6345,
                "stable_mae": 23.3237,
            },
            {
                "baseline_family": "dwellnet_validated",
                "method": "dwell_mlp_dwell_supervised_residual_dwell_wc8_branchwise",
                "dwell_mae": 35.6457,
                "unobserved_mae": 40.7380,
                "weighted_energy_error": 0.0145,
                "transition_mae": 77.2218,
                "stable_mae": 22.6749,
            },
            {
                "baseline_family": "dwellnet_secondary",
                "method": "dwell_transformer_dwell_energy_residual_dwell_wc8_branchwise_notime",
                "dwell_mae": 35.1171,
                "unobserved_mae": 40.1338,
                "weighted_energy_error": 0.0124,
                "transition_mae": 72.9469,
                "stable_mae": 23.3151,
            },
        ]
    )


def compare_observer_to_references(leaderboard: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baselines = reference_baselines()
    transition_by_method = transitions.set_index("method") if not transitions.empty else pd.DataFrame()
    for observer in leaderboard.itertuples(index=False):
        observer_transition = (
            float(transition_by_method.loc[observer.method, "transition_mae"])
            if not transition_by_method.empty and observer.method in transition_by_method.index
            else float("nan")
        )
        for baseline in baselines.itertuples(index=False):
            rows.append(
                {
                    "observer_method": observer.method,
                    "baseline_family": baseline.baseline_family,
                    "baseline_method": baseline.method,
                    "observer_dwell_mae": observer.dwell_mae,
                    "baseline_dwell_mae": baseline.dwell_mae,
                    "dwell_mae_delta": observer.dwell_mae - baseline.dwell_mae,
                    "observer_unobserved_mae": observer.unobserved_point_mae,
                    "baseline_unobserved_mae": baseline.unobserved_mae,
                    "unobserved_mae_delta": observer.unobserved_point_mae - baseline.unobserved_mae,
                    "observer_weighted_energy_error": observer.weighted_energy_error,
                    "baseline_weighted_energy_error": baseline.weighted_energy_error,
                    "energy_error_delta": observer.weighted_energy_error - baseline.weighted_energy_error,
                    "observer_transition_mae": observer_transition,
                    "baseline_transition_mae": baseline.transition_mae,
                    "transition_mae_delta": observer_transition - baseline.transition_mae
                    if np.isfinite(baseline.transition_mae) and np.isfinite(observer_transition)
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)
