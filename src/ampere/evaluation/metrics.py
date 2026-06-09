from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EPSILON_WH = 1e-9
NEAR_ZERO_POWER_W = 1.0
BASE_GROUP_COLUMNS = ["dataset_id", "method"]
OPTIONAL_GROUP_COLUMNS = [
    "reconstruction_mode",
    "target_mode",
    "model_family",
    "model_level",
    "neural_architecture",
    "loss_variant",
    "original_method",
    "constraint_method",
    "window_cycles",
    "output_mode",
    "normalization_mode",
    "use_branch_embeddings",
    "use_time_features",
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


@dataclass(frozen=True)
class MetricTables:
    overall: pd.DataFrame
    branch: pd.DataFrame
    dwell: pd.DataFrame


def _safe_mean(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def _pointwise_metrics(group: pd.DataFrame) -> dict[str, float]:
    error = group["P_hat"].to_numpy(dtype=float) - group["P_true"].to_numpy(dtype=float)
    abs_error = np.abs(error)
    return {
        "n": int(len(group)),
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "median_abs_error": float(np.median(abs_error)),
        "max_abs_error": float(np.max(abs_error)),
        "signed_bias": float(np.mean(error)),
    }


def _energy_metrics(group: pd.DataFrame) -> dict[str, float]:
    dt = group["dt_s"].to_numpy(dtype=float)
    true = group["P_true"].to_numpy(dtype=float)
    hat = group["P_hat"].to_numpy(dtype=float)
    e_true = float(np.sum(true * dt) / 3600.0)
    e_hat = float(np.sum(hat * dt) / 3600.0)
    e_true_abs = float(np.sum(np.abs(true) * dt) / 3600.0)
    e_true_clip = float(np.sum(np.maximum(true, 0.0) * dt) / 3600.0)
    e_hat_clip = float(np.sum(np.maximum(hat, 0.0) * dt) / 3600.0)
    negative_energy = float(np.sum(np.abs(np.minimum(true, 0.0)) * dt) / 3600.0)
    return {
        "E_true_signed_Wh": e_true,
        "E_hat_signed_Wh": e_hat,
        "abs_energy_error_Wh": abs(e_hat - e_true),
        "E_true_abs_Wh": e_true_abs,
        "E_true_clipped_nonnegative_diagnostic_Wh": e_true_clip,
        "E_hat_clipped_nonnegative_diagnostic_Wh": e_hat_clip,
        "abs_energy_error_clipped_nonnegative_diagnostic_Wh": abs(e_hat_clip - e_true_clip),
        "branch_negative_energy_fraction": negative_energy / (EPSILON_WH + e_true_abs),
    }


def _observation_metrics(group: pd.DataFrame) -> dict[str, float]:
    observed = group["is_observed"].to_numpy(dtype=bool)
    error = np.abs(group["P_hat"].to_numpy(dtype=float) - group["P_true"].to_numpy(dtype=float))
    return {
        "observed_point_mae": _safe_mean(error[observed]),
        "unobserved_point_mae": _safe_mean(error[~observed]),
    }


def _activity_metrics(group: pd.DataFrame, near_zero_threshold: float) -> dict[str, float]:
    true = group["P_true"].to_numpy(dtype=float)
    hat = group["P_hat"].to_numpy(dtype=float)
    near_zero = np.abs(true) <= near_zero_threshold
    negative = true < 0
    return {
        "fraction_near_zero_truth": float(np.mean(near_zero)),
        "off_state_false_positive_power": _safe_mean(np.abs(hat[near_zero])),
        "branch_negative_fraction": float(np.mean(negative)),
    }


def metric_group_columns(df: pd.DataFrame, *, include_branch: bool = False) -> list[str]:
    columns = list(BASE_GROUP_COLUMNS)
    columns.extend([column for column in OPTIONAL_GROUP_COLUMNS if column in df.columns])
    if include_branch:
        columns.extend(["branch_id", "branch_name"])
    return columns


def _key_dict(columns: list[str], keys: object) -> dict[str, object]:
    if len(columns) == 1:
        values = (keys,)
    else:
        values = keys
    return dict(zip(columns, values))


def compute_dwell_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset_id", "method", "branch_id", "branch_name", "dwell_id", "P_true", "P_hat", "dt_s"}
    if not required.issubset(predictions.columns):
        return pd.DataFrame()

    branch_cols = metric_group_columns(predictions, include_branch=True)
    group_cols = branch_cols + ["dwell_id"]
    tmp = predictions[list(group_cols) + ["P_true", "P_hat", "dt_s"]].copy()
    tmp["true_Wh"] = tmp["P_true"] * tmp["dt_s"] / 3600.0
    tmp["hat_Wh"] = tmp["P_hat"] * tmp["dt_s"] / 3600.0
    dwell = (
        tmp.groupby(group_cols, sort=False)
        .agg(
            P_true_dwell_mean=("P_true", "mean"),
            P_hat_dwell_mean=("P_hat", "mean"),
            E_true_dwell_Wh=("true_Wh", "sum"),
            E_hat_dwell_Wh=("hat_Wh", "sum"),
        )
        .reset_index()
    )
    dwell["dwell_abs_error"] = np.abs(dwell["P_hat_dwell_mean"] - dwell["P_true_dwell_mean"])
    dwell["dwell_energy_abs_error_Wh"] = np.abs(dwell["E_hat_dwell_Wh"] - dwell["E_true_dwell_Wh"])

    summary = (
        dwell.assign(dwell_sq_error=dwell["dwell_abs_error"] ** 2)
        .groupby(branch_cols, sort=False)
        .agg(
            dwell_count=("dwell_id", "count"),
            dwell_mae=("dwell_abs_error", "mean"),
            dwell_sq_error_mean=("dwell_sq_error", "mean"),
            dwell_abs_energy_error_Wh_total=("dwell_energy_abs_error_Wh", "sum"),
        )
        .reset_index()
    )
    summary["dwell_rmse"] = np.sqrt(summary["dwell_sq_error_mean"])
    return summary.drop(columns=["dwell_sq_error_mean"])


def compute_branch_metrics(
    predictions: pd.DataFrame,
    *,
    near_zero_threshold: float = NEAR_ZERO_POWER_W,
) -> pd.DataFrame:
    rows = []
    group_cols = metric_group_columns(predictions, include_branch=True)
    for keys, group in predictions.groupby(group_cols, sort=False):
        row = _key_dict(group_cols, keys)
        row["branch_id"] = int(row["branch_id"])
        row.update(_pointwise_metrics(group))
        row.update(_energy_metrics(group))
        row.update(_observation_metrics(group))
        row.update(_activity_metrics(group, near_zero_threshold))
        rows.append(row)

    branch = pd.DataFrame(rows)
    dwell = compute_dwell_metrics(predictions)
    if not dwell.empty:
        branch = branch.merge(
            dwell,
            on=group_cols,
            how="left",
        )
    return branch


def compute_overall_metrics(predictions: pd.DataFrame, branch_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = metric_group_columns(predictions, include_branch=False)
    for keys, group in predictions.groupby(group_cols, sort=False):
        row = _key_dict(group_cols, keys)
        row.update(_pointwise_metrics(group))
        row.update(_observation_metrics(group))
        row.update(_activity_metrics(group, NEAR_ZERO_POWER_W))
        branch_mask = pd.Series(True, index=branch_metrics.index)
        for column in group_cols:
            branch_mask &= branch_metrics[column] == row[column]
        branch_subset = branch_metrics[branch_mask]
        e_true_abs_sum = float(np.sum(np.abs(branch_subset["E_true_signed_Wh"].to_numpy(dtype=float))))
        e_error_sum = float(np.sum(branch_subset["abs_energy_error_Wh"].to_numpy(dtype=float)))
        e_true_clip_abs_sum = float(
            np.sum(np.abs(branch_subset["E_true_clipped_nonnegative_diagnostic_Wh"].to_numpy(dtype=float)))
        )
        e_error_clip_sum = float(
            np.sum(branch_subset["abs_energy_error_clipped_nonnegative_diagnostic_Wh"].to_numpy(dtype=float))
        )
        row.update(
            {
                "branch_count": int(branch_subset["branch_id"].nunique()),
                "E_true_signed_Wh_sum": float(branch_subset["E_true_signed_Wh"].sum()),
                "E_hat_signed_Wh_sum": float(branch_subset["E_hat_signed_Wh"].sum()),
                "abs_energy_error_Wh_sum": e_error_sum,
                "weighted_energy_error": e_error_sum / (EPSILON_WH + e_true_abs_sum),
                "weighted_energy_error_clipped_nonnegative_diagnostic": e_error_clip_sum
                / (EPSILON_WH + e_true_clip_abs_sum),
                "dwell_mae": float(branch_subset["dwell_mae"].mean())
                if "dwell_mae" in branch_subset.columns
                else float("nan"),
                "dwell_abs_energy_error_Wh_total": float(branch_subset["dwell_abs_energy_error_Wh_total"].sum())
                if "dwell_abs_energy_error_Wh_total" in branch_subset.columns
                else float("nan"),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_predictions(predictions: pd.DataFrame) -> MetricTables:
    branch = compute_branch_metrics(predictions)
    overall = compute_overall_metrics(predictions, branch)
    dwell = compute_dwell_metrics(predictions)
    return MetricTables(overall=overall, branch=branch, dwell=dwell)
