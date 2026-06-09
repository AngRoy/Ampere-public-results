from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ampere.features.tabular import make_feature_frame, target_column_for


WINDOW_KEYS = ["dataset_id", "branch_id", "dwell_id"]
WINDOW_META_COLUMNS = [
    "dataset_id",
    "source_type",
    "time_s",
    "time_index",
    "branch_id",
    "branch_name",
    "load_type",
    "selected_branch",
    "dwell_id",
    "scan_cycle_id",
    "dt_s",
    "dwell_s_requested",
    "dwell_s_effective",
    "time_axis_confidence",
    "scaling_applied",
    "source_file",
    "split",
]


@dataclass(frozen=True)
class WindowFeatureMatrix:
    window_frame: pd.DataFrame
    X: pd.DataFrame
    y: np.ndarray
    feature_columns: list[str]
    target_column: str
    target_mode: str
    reconstruction_mode: str


def make_window_features(
    df: pd.DataFrame,
    *,
    reconstruction_mode: str,
    target_mode: str,
    fit_columns: list[str] | None = None,
) -> WindowFeatureMatrix:
    row_features = make_feature_frame(df, reconstruction_mode=reconstruction_mode)
    grouping = [df[column] for column in WINDOW_KEYS]
    X = row_features.groupby(grouping, sort=False).mean().reset_index(drop=True)

    meta_columns = [column for column in WINDOW_META_COLUMNS if column in df.columns]
    meta = df.groupby(WINDOW_KEYS, sort=False)[meta_columns].first().reset_index(drop=True)
    target_column = target_column_for(target_mode)
    if target_column not in df.columns:
        raise ValueError(f"Window target requires column {target_column}")
    y = df.groupby(WINDOW_KEYS, sort=False)[target_column].mean().to_numpy(dtype=float)
    meta["window_target"] = y

    if fit_columns is not None:
        X = X.reindex(columns=fit_columns, fill_value=0.0)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return WindowFeatureMatrix(
        window_frame=meta,
        X=X,
        y=y,
        feature_columns=list(X.columns),
        target_column=target_column,
        target_mode=target_mode,
        reconstruction_mode=reconstruction_mode,
    )
