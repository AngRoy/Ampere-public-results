from __future__ import annotations

import numpy as np
import pandas as pd

from ampere.neural.datasets import NeuralSeries


NEURAL_PREDICTION_COLUMNS = [
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
]


def build_neural_prediction_frame(
    series: NeuralSeries,
    p_hat_matrix: np.ndarray,
    *,
    loss_variant: str,
    split: str = "test",
    neural_architecture: str = "causal_tcn",
    hard_observed_projection: bool = True,
) -> pd.DataFrame:
    target_column = "P_true" if series.target_mode == "raw_signed_power" else "P_dwell_mean"
    frame = series.long[series.long["split"].eq(split)].copy()
    branch_to_pos = {branch_id: idx for idx, branch_id in enumerate(series.branch_ids)}
    time_to_pos = {int(time_index): idx for idx, time_index in enumerate(series.time_index)}
    frame["P_true"] = frame[target_column].to_numpy(dtype=float)
    frame["power_representation"] = series.target_mode
    rows = frame[["time_index", "branch_id"]].to_numpy(dtype=np.int64)
    frame["P_hat"] = pd.Series(
        [float(p_hat_matrix[time_to_pos[int(t)], branch_to_pos[int(b)]]) for t, b in rows],
        index=frame.index,
        dtype=float,
    )

    if hard_observed_projection and series.target_mode == "raw_signed_power":
        observed = frame["is_observed"].astype(bool) & frame["P_observed"].notna()
        frame.loc[observed, "P_hat"] = frame.loc[observed, "P_observed"]

    method = f"causal_tcn_{loss_variant}"
    frame["method"] = method
    frame["model_family"] = "neural"
    frame["neural_architecture"] = neural_architecture
    frame["model_level"] = "window_sequence"
    frame["reconstruction_mode"] = series.reconstruction_mode
    frame["target_mode"] = series.target_mode
    frame["loss_variant"] = loss_variant

    optional = [
        "model_level",
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
    columns = NEURAL_PREDICTION_COLUMNS + [column for column in optional if column in frame.columns]
    return frame[columns]


def validate_neural_prediction_schema(predictions: pd.DataFrame) -> None:
    missing = [column for column in NEURAL_PREDICTION_COLUMNS if column not in predictions.columns]
    if missing:
        raise AssertionError(f"Neural predictions missing required columns: {missing}")
    key = [
        "dataset_id",
        "reconstruction_mode",
        "target_mode",
        "loss_variant",
        "time_index",
        "branch_id",
    ]
    duplicates = int(predictions.duplicated(key).sum())
    if duplicates:
        raise AssertionError(f"Neural predictions contain duplicate key rows: {duplicates}")
    if predictions["power_representation"].isna().any():
        raise AssertionError("power_representation contains nulls")
    if predictions["P_raw_signed"].isna().any():
        raise AssertionError("P_raw_signed contains nulls")


def build_loss_ablation(leaderboard: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset_id", "target_mode", "reconstruction_mode"]
    for keys, group in leaderboard.groupby(group_cols, sort=False):
        base = group[group["loss_variant"].eq("neural_base")]
        if base.empty:
            continue
        base_row = base.sort_values("mae").iloc[0]
        for row in group.itertuples(index=False):
            row_dict = row._asdict()
            rows.append(
                {
                    "dataset_id": keys[0],
                    "target_mode": keys[1],
                    "reconstruction_mode": keys[2],
                    "loss_variant": row_dict["loss_variant"],
                    "method": row_dict["method"],
                    "mae": row_dict["mae"],
                    "dwell_mae": row_dict["dwell_mae"],
                    "weighted_energy_error": row_dict["weighted_energy_error"],
                    "mae_delta_vs_neural_base": row_dict["mae"] - base_row["mae"],
                    "dwell_mae_delta_vs_neural_base": row_dict["dwell_mae"] - base_row["dwell_mae"],
                    "weighted_energy_error_delta_vs_neural_base": row_dict["weighted_energy_error"]
                    - base_row["weighted_energy_error"],
                }
            )
    return pd.DataFrame(rows)
