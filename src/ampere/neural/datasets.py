from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ampere.features.splits import add_time_block_split
from ampere.features.tabular import target_column_for
from ampere.neural.utils import PowerScaler, require_torch


POWER_TARGET_MODES = ("raw_signed_power", "dwell_mean_power")
RECONSTRUCTION_MODES = ("online_safe", "offline")
ONLINE_SAFE_FORBIDDEN_FEATURES = {
    "next_observed_value",
    "time_to_next_seen",
    "next_observed_available",
}


@dataclass(frozen=True)
class NeuralSeries:
    dataset_id: str
    source_type: str
    target_mode: str
    reconstruction_mode: str
    branch_ids: list[int]
    branch_names: list[str]
    time_index: np.ndarray
    time_s: np.ndarray
    split: np.ndarray
    dt_s: np.ndarray
    dwell_id: np.ndarray
    dwell_position: np.ndarray
    selected_branch: np.ndarray
    features: np.ndarray
    feature_names: list[str]
    target_scaled: np.ndarray
    target_physical: np.ndarray
    raw_signed: np.ndarray
    observed_values_scaled: np.ndarray
    observed_values_physical: np.ndarray
    observed_mask: np.ndarray
    scaler: PowerScaler
    long: pd.DataFrame

    @property
    def branch_count(self) -> int:
        return len(self.branch_ids)

    @property
    def feature_count(self) -> int:
        return int(self.features.shape[1])


def _pivot_matrix(df: pd.DataFrame, value_column: str, branch_ids: list[int]) -> np.ndarray:
    matrix = (
        df.pivot(index="time_index", columns="branch_id", values=value_column)
        .sort_index()
        .reindex(columns=branch_ids)
        .to_numpy(dtype=float)
    )
    return matrix


def _time_meta(df: pd.DataFrame) -> pd.DataFrame:
    meta_columns = [
        "dataset_id",
        "source_type",
        "time_s",
        "time_index",
        "selected_branch",
        "dwell_id",
        "scan_cycle_id",
        "dwell_position",
        "dt_s",
        "dwell_s_effective",
        "split",
    ]
    present = [column for column in meta_columns if column in df.columns]
    return df.groupby("time_index", sort=True)[present].first().reset_index(drop=True)


def _scale_power_feature(values: np.ndarray, scaler: PowerScaler, *, fill_value: float = 0.0) -> np.ndarray:
    scaled = scaler.transform_numpy(values)
    return np.where(np.isfinite(scaled), scaled, fill_value).astype(np.float32)


def _normalize_time_since(values: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    train_values = values[train_mask]
    finite = train_values[np.isfinite(train_values)]
    denom = float(np.nanmax(finite)) if len(finite) else 1.0
    if not np.isfinite(denom) or denom <= 0:
        denom = 1.0
    normalized = np.where(np.isfinite(values), values / denom, 1.0)
    return np.clip(normalized, 0.0, 5.0).astype(np.float32)


def _selected_onehot(selected_branch: np.ndarray, branch_ids: list[int]) -> np.ndarray:
    result = np.zeros((len(selected_branch), len(branch_ids)), dtype=np.float32)
    branch_to_pos = {branch_id: idx for idx, branch_id in enumerate(branch_ids)}
    for row, branch in enumerate(selected_branch):
        pos = branch_to_pos.get(int(branch))
        if pos is not None:
            result[row, pos] = 1.0
    return result


def build_neural_series(
    long_df: pd.DataFrame,
    *,
    target_mode: str,
    reconstruction_mode: str = "online_safe",
) -> NeuralSeries:
    if target_mode not in POWER_TARGET_MODES:
        raise ValueError(f"Unknown target mode: {target_mode}")
    if reconstruction_mode not in RECONSTRUCTION_MODES:
        raise ValueError(f"Unknown reconstruction mode: {reconstruction_mode}")

    df = long_df.copy()
    if "split" not in df.columns:
        df = add_time_block_split(df)

    dataset_ids = sorted(df["dataset_id"].dropna().unique().tolist())
    if len(dataset_ids) != 1:
        raise ValueError(f"NeuralSeries expects one dataset_id, got {dataset_ids}")
    dataset_id = str(dataset_ids[0])
    source_type = str(df["source_type"].dropna().iloc[0]) if "source_type" in df.columns else "unknown"
    branch_ids = [int(value) for value in sorted(df["branch_id"].dropna().unique().tolist())]
    branch_names = (
        df[["branch_id", "branch_name"]]
        .drop_duplicates("branch_id")
        .sort_values("branch_id")["branch_name"]
        .astype(str)
        .tolist()
    )

    target_column = target_column_for(target_mode)
    target = _pivot_matrix(df, target_column, branch_ids)
    raw_signed = _pivot_matrix(df, "P_raw_signed", branch_ids)
    observed_physical = _pivot_matrix(df, "P_observed", branch_ids)
    observed_mask = np.isfinite(observed_physical).astype(np.float32)
    last_observed = _pivot_matrix(df, "last_observed_value", branch_ids)
    time_since = _pivot_matrix(df, "time_since_last_seen", branch_ids)

    meta = _time_meta(df)
    split = meta["split"].astype(str).to_numpy()
    train_time_mask = split == "train"
    train_targets = target[train_time_mask]
    scaler = PowerScaler.fit(train_targets)

    target_scaled = _scale_power_feature(target, scaler)
    observed_scaled = _scale_power_feature(observed_physical, scaler)
    last_scaled = _scale_power_feature(last_observed, scaler)
    last_available = np.isfinite(last_observed).astype(np.float32)
    time_since_norm = _normalize_time_since(time_since, train_time_mask)
    selected = meta["selected_branch"].to_numpy(dtype=np.int64)
    selected_onehot = _selected_onehot(selected, branch_ids)

    feature_blocks: list[np.ndarray] = [
        observed_scaled,
        observed_mask,
        last_scaled,
        last_available,
        time_since_norm,
        selected_onehot,
    ]
    feature_names = (
        [f"observed_value_b{branch_id:02d}" for branch_id in branch_ids]
        + [f"observed_mask_b{branch_id:02d}" for branch_id in branch_ids]
        + [f"last_observed_value_b{branch_id:02d}" for branch_id in branch_ids]
        + [f"last_observed_available_b{branch_id:02d}" for branch_id in branch_ids]
        + [f"time_since_last_seen_b{branch_id:02d}" for branch_id in branch_ids]
        + [f"selected_branch_b{branch_id:02d}" for branch_id in branch_ids]
    )

    if reconstruction_mode == "offline":
        next_observed = _pivot_matrix(df, "next_observed_value", branch_ids)
        time_to_next = _pivot_matrix(df, "time_to_next_seen", branch_ids)
        next_scaled = _scale_power_feature(next_observed, scaler)
        next_available = np.isfinite(next_observed).astype(np.float32)
        time_to_next_norm = _normalize_time_since(time_to_next, train_time_mask)
        feature_blocks.extend([next_scaled, next_available, time_to_next_norm])
        feature_names.extend(
            [f"next_observed_value_b{branch_id:02d}" for branch_id in branch_ids]
            + [f"next_observed_available_b{branch_id:02d}" for branch_id in branch_ids]
            + [f"time_to_next_seen_b{branch_id:02d}" for branch_id in branch_ids]
        )

    dwell_effective = meta.get("dwell_s_effective", pd.Series(1.0, index=meta.index)).replace(0, np.nan)
    dwell_position_norm = (
        meta["dwell_position"].astype(float) * meta["dt_s"].astype(float) / dwell_effective.astype(float)
    ).fillna(0.0)
    feature_blocks.append(dwell_position_norm.to_numpy(dtype=np.float32).reshape(-1, 1))
    feature_names.append("dwell_position_norm")

    features = np.concatenate(feature_blocks, axis=1).astype(np.float32)
    if reconstruction_mode == "online_safe":
        forbidden = ONLINE_SAFE_FORBIDDEN_FEATURES & set(feature_names)
        if forbidden:
            raise ValueError(f"Online-safe neural features contain future context: {sorted(forbidden)}")

    return NeuralSeries(
        dataset_id=dataset_id,
        source_type=source_type,
        target_mode=target_mode,
        reconstruction_mode=reconstruction_mode,
        branch_ids=branch_ids,
        branch_names=branch_names,
        time_index=meta["time_index"].to_numpy(dtype=np.int64),
        time_s=meta["time_s"].to_numpy(dtype=float),
        split=split,
        dt_s=meta["dt_s"].to_numpy(dtype=float),
        dwell_id=meta["dwell_id"].to_numpy(dtype=np.int64),
        dwell_position=meta["dwell_position"].to_numpy(dtype=np.int64),
        selected_branch=selected,
        features=features,
        feature_names=feature_names,
        target_scaled=target_scaled.astype(np.float32),
        target_physical=target.astype(np.float32),
        raw_signed=raw_signed.astype(np.float32),
        observed_values_scaled=observed_scaled.astype(np.float32),
        observed_values_physical=np.where(np.isfinite(observed_physical), observed_physical, np.nan).astype(np.float32),
        observed_mask=observed_mask.astype(np.float32),
        scaler=scaler,
        long=df,
    )


def window_starts_for_split(
    series: NeuralSeries,
    *,
    split: str,
    window_size: int,
    stride: int,
    max_windows: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    positions = np.flatnonzero(series.split == split)
    if len(positions) == 0:
        raise ValueError(f"No positions available for split {split}")
    start_min = int(positions.min())
    start_max = int(positions.max()) - window_size + 1
    if start_max < start_min:
        raise ValueError(f"Split {split} is shorter than window_size={window_size}")
    starts = np.arange(start_min, start_max + 1, stride, dtype=np.int64)
    final_start = np.int64(start_max)
    if len(starts) == 0 or starts[-1] != final_start:
        starts = np.concatenate([starts, np.asarray([final_start], dtype=np.int64)])

    if max_windows is not None and len(starts) > max_windows:
        rng = np.random.default_rng(seed)
        starts = np.sort(rng.choice(starts, size=max_windows, replace=False))
    return starts.astype(np.int64)


class WindowedReconstructionDataset:
    def __init__(self, series: NeuralSeries, starts: np.ndarray, window_size: int):
        self.series = series
        self.starts = np.asarray(starts, dtype=np.int64)
        self.window_size = int(window_size)
        self._torch = require_torch()

    def __len__(self) -> int:
        return int(len(self.starts))

    def __getitem__(self, index: int) -> dict[str, object]:
        start = int(self.starts[index])
        stop = start + self.window_size
        torch = self._torch
        return {
            "x": torch.as_tensor(self.series.features[start:stop].T, dtype=torch.float32),
            "target": torch.as_tensor(self.series.target_scaled[start:stop].T, dtype=torch.float32),
            "observed_values": torch.as_tensor(self.series.observed_values_scaled[start:stop].T, dtype=torch.float32),
            "observed_mask": torch.as_tensor(self.series.observed_mask[start:stop].T, dtype=torch.float32),
            "dt_s": torch.as_tensor(self.series.dt_s[start:stop], dtype=torch.float32),
            "dwell_id": torch.as_tensor(self.series.dwell_id[start:stop], dtype=torch.long),
            "start": torch.as_tensor(start, dtype=torch.long),
        }
