from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from ampere.features.splits import add_time_block_split
from ampere.neural.utils import require_torch


DWELL_TARGET_MODES = ("dwell_mean_power", "raw_signed_power")
OUTPUT_MODES = ("absolute", "residual_dwell")
NORMALIZATION_MODES = ("branchwise", "global")
DATASET_SPECS = {
    "appliance_8ch": {"branch_count": 8, "dwell_samples": 1, "scan_cycle_samples": 8, "window_cycles": (4, 8, 16)},
    "rlc_sample": {"branch_count": 4, "dwell_samples": 25, "scan_cycle_samples": 100, "window_cycles": (1, 2, 4)},
}
FORBIDDEN_DWELL_FEATURE_TOKENS = ("next_observed", "time_to_next", "root_power")


@dataclass(frozen=True)
class DwellNormalizer:
    mode: str
    mean: np.ndarray
    std: np.ndarray
    branch_ids: list[int]
    value_kind: str

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        *,
        train_mask: np.ndarray,
        branch_ids: list[int],
        mode: str = "branchwise",
        value_kind: str = "target",
    ) -> "DwellNormalizer":
        if mode not in NORMALIZATION_MODES:
            raise ValueError(f"Unknown normalization mode: {mode}")
        train_values = np.asarray(values, dtype=float)[train_mask]
        if mode == "global":
            finite = train_values[np.isfinite(train_values)]
            mean_value = float(np.mean(finite)) if len(finite) else 0.0
            std_value = float(np.std(finite)) if len(finite) else 1.0
            if not np.isfinite(std_value) or std_value < 1e-9:
                std_value = 1.0
            mean = np.full(len(branch_ids), mean_value, dtype=float)
            std = np.full(len(branch_ids), std_value, dtype=float)
        else:
            mean = np.zeros(len(branch_ids), dtype=float)
            std = np.ones(len(branch_ids), dtype=float)
            for idx in range(len(branch_ids)):
                finite = train_values[:, idx]
                finite = finite[np.isfinite(finite)]
                if len(finite):
                    mean[idx] = float(np.mean(finite))
                    std[idx] = float(np.std(finite))
                if not np.isfinite(std[idx]) or std[idx] < 1e-9:
                    std[idx] = 1.0
        return cls(mode=mode, mean=mean, std=std, branch_ids=branch_ids, value_kind=value_kind)

    def transform(self, values: np.ndarray, *, fill_value: float = 0.0) -> np.ndarray:
        scaled = (np.asarray(values, dtype=float) - self.mean.reshape(1, -1)) / self.std.reshape(1, -1)
        return np.where(np.isfinite(scaled), scaled, fill_value).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.std.reshape(1, -1) + self.mean.reshape(1, -1)

    def inverse_torch(self, values):
        torch = require_torch()
        mean = torch.as_tensor(self.mean, dtype=values.dtype, device=values.device).view(1, -1)
        std = torch.as_tensor(self.std, dtype=values.dtype, device=values.device).view(1, -1)
        return values * std + mean

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "branch_ids": self.branch_ids,
            "value_kind": self.value_kind,
        }


@dataclass(frozen=True)
class DwellSeries:
    dataset_id: str
    source_type: str
    target_mode: str
    reconstruction_mode: str
    output_mode: str
    normalization_mode: str
    include_time_features: bool
    include_branch_static: bool
    branch_ids: list[int]
    branch_names: list[str]
    load_types: list[str]
    dwell_id: np.ndarray
    scan_cycle_id: np.ndarray
    cycle_position: np.ndarray
    selected_branch: np.ndarray
    split: np.ndarray
    time_s: np.ndarray
    time_index_start: np.ndarray
    time_index_end: np.ndarray
    dt_s: np.ndarray
    dwell_s_effective: np.ndarray
    target_physical: np.ndarray
    target_scaled: np.ndarray
    output_target_scaled: np.ndarray
    residual_physical: np.ndarray
    observed_values_physical: np.ndarray
    observed_mask: np.ndarray
    last_observed_physical: np.ndarray
    last_observed_scaled: np.ndarray
    last_observed_available: np.ndarray
    time_since_last_seen: np.ndarray
    features: np.ndarray
    feature_names: list[str]
    target_normalizer: DwellNormalizer
    output_normalizer: DwellNormalizer
    token_table: pd.DataFrame
    long: pd.DataFrame

    @property
    def branch_count(self) -> int:
        return len(self.branch_ids)

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[2])


def dwell_target_column_for(target_mode: str) -> str:
    if target_mode not in DWELL_TARGET_MODES:
        raise ValueError(f"Unknown DwellNet target mode: {target_mode}")
    return "P_dwell_mean"


def dataset_spec(dataset_id: str) -> dict[str, Any]:
    if dataset_id not in DATASET_SPECS:
        raise ValueError(f"Unsupported DwellNet dataset: {dataset_id}")
    return DATASET_SPECS[dataset_id]


def _safe_normalize(values: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    train = np.asarray(values, dtype=float)[train_mask]
    finite = train[np.isfinite(train)]
    if len(finite) == 0:
        return np.zeros_like(values, dtype=np.float32)
    min_value = float(np.min(finite))
    max_value = float(np.max(finite))
    span = max_value - min_value
    if not np.isfinite(span) or span <= 0:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((np.asarray(values, dtype=float) - min_value) / span, 0.0, 1.0).astype(np.float32)


def _mode_or_first(values: pd.Series) -> object:
    mode = values.mode(dropna=True)
    if not mode.empty:
        return mode.iloc[0]
    return values.iloc[0]


def _pivot_dwell_branch(df: pd.DataFrame, value_column: str, branch_ids: list[int], agg: str = "mean") -> np.ndarray:
    grouped = df.groupby(["dwell_id", "branch_id"], sort=True)[value_column]
    values = grouped.mean() if agg == "mean" else grouped.first()
    return (
        values.reset_index()
        .pivot(index="dwell_id", columns="branch_id", values=value_column)
        .sort_index()
        .reindex(columns=branch_ids)
        .to_numpy(dtype=float)
    )


def build_dwell_token_table(long_df: pd.DataFrame, *, target_mode: str = "dwell_mean_power") -> pd.DataFrame:
    df = long_df.copy()
    if "split" not in df.columns:
        df = add_time_block_split(df)
    required = {
        "dataset_id",
        "source_type",
        "time_s",
        "time_index",
        "branch_id",
        "branch_name",
        "load_type",
        "P_dwell_mean",
        "P_observed",
        "is_observed",
        "selected_branch",
        "dwell_id",
        "scan_cycle_id",
        "dt_s",
        "split",
        "dwell_s_effective",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Cannot build dwell table; missing columns: {missing}")

    dataset_ids = sorted(df["dataset_id"].dropna().unique().tolist())
    if len(dataset_ids) != 1:
        raise ValueError(f"Expected one dataset_id, got {dataset_ids}")
    dataset_id = str(dataset_ids[0])
    spec = dataset_spec(dataset_id)
    branch_ids = [int(value) for value in sorted(df["branch_id"].dropna().unique().tolist())]
    if len(branch_ids) != int(spec["branch_count"]):
        raise ValueError(f"{dataset_id} expected {spec['branch_count']} branches, found {len(branch_ids)}")

    target_column = dwell_target_column_for(target_mode)
    target = _pivot_dwell_branch(df, target_column, branch_ids, agg="mean")
    observed = _pivot_dwell_branch(df, "P_observed", branch_ids, agg="mean")
    observed_mask = np.isfinite(observed)

    meta = (
        df.groupby("dwell_id", sort=True)
        .agg(
            dataset_id=("dataset_id", "first"),
            source_type=("source_type", "first"),
            time_s=("time_s", "max"),
            time_index_start=("time_index", "min"),
            time_index_end=("time_index", "max"),
            selected_branch=("selected_branch", "first"),
            scan_cycle_id=("scan_cycle_id", "first"),
            dt_s=("dt_s", "first"),
            dwell_s_effective=("dwell_s_effective", "first"),
            split=("split", _mode_or_first),
        )
        .reset_index()
        .sort_values("dwell_id")
    )
    meta["cycle_position"] = meta["dwell_id"].astype(int) % len(branch_ids)

    table = meta.copy()
    for idx, branch_id in enumerate(branch_ids):
        table[f"target_b{branch_id:02d}"] = target[:, idx]
        table[f"observed_b{branch_id:02d}"] = observed[:, idx]
        table[f"observed_mask_b{branch_id:02d}"] = observed_mask[:, idx]
    return table


def _branch_metadata(df: pd.DataFrame, branch_ids: list[int]) -> tuple[list[str], list[str]]:
    meta = df[["branch_id", "branch_name", "load_type"]].drop_duplicates("branch_id").sort_values("branch_id")
    meta = meta.set_index("branch_id").reindex(branch_ids)
    return meta["branch_name"].astype(str).tolist(), meta["load_type"].astype(str).tolist()


def _post_dwell_last_observed(observed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    token_count, branch_count = observed.shape
    last_values = np.full((token_count, branch_count), np.nan, dtype=float)
    last_available = np.zeros((token_count, branch_count), dtype=bool)
    time_since = np.full((token_count, branch_count), np.nan, dtype=float)
    current = np.full(branch_count, np.nan, dtype=float)
    seen_at = np.full(branch_count, -1, dtype=int)
    for token_idx in range(token_count):
        observed_now = np.isfinite(observed[token_idx])
        current[observed_now] = observed[token_idx, observed_now]
        seen_at[observed_now] = token_idx
        last_values[token_idx] = current
        available = seen_at >= 0
        last_available[token_idx] = available
        time_since[token_idx, available] = token_idx - seen_at[available]
    return last_values, last_available, time_since


def _static_branch_features(
    branch_ids: list[int],
    load_types: list[str],
    token_count: int,
    *,
    include_branch_static: bool,
) -> tuple[np.ndarray, list[str]]:
    branch_count = len(branch_ids)
    branch_pos = (np.arange(branch_count, dtype=np.float32) / max(1, branch_count - 1)).reshape(1, branch_count, 1)
    blocks = [np.repeat(branch_pos, token_count, axis=0)]
    names = ["branch_position_norm"]
    if include_branch_static:
        branch_eye = np.eye(branch_count, dtype=np.float32).reshape(1, branch_count, branch_count)
        blocks.append(np.repeat(branch_eye, token_count, axis=0))
        names.extend([f"branch_identity_b{branch_id:02d}" for branch_id in branch_ids])
        load_categories = sorted(set(load_types))
        load_index = {value: idx for idx, value in enumerate(load_categories)}
        load = np.zeros((branch_count, len(load_categories)), dtype=np.float32)
        for branch_idx, load_type in enumerate(load_types):
            load[branch_idx, load_index[load_type]] = 1.0
        blocks.append(np.repeat(load.reshape(1, branch_count, len(load_categories)), token_count, axis=0))
        names.extend([f"load_type_{value}" for value in load_categories])
    return np.concatenate(blocks, axis=2), names


def build_dwell_series(
    long_df: pd.DataFrame,
    *,
    target_mode: str = "dwell_mean_power",
    reconstruction_mode: str = "online_safe",
    output_mode: str = "residual_dwell",
    normalization_mode: str = "branchwise",
    include_time_features: bool = True,
    include_branch_static: bool = True,
) -> DwellSeries:
    if reconstruction_mode != "online_safe":
        raise ValueError("DwellNet currently implements the post-dwell online_safe task only")
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f"Unknown DwellNet output mode: {output_mode}")
    if normalization_mode not in NORMALIZATION_MODES:
        raise ValueError(f"Unknown normalization mode: {normalization_mode}")

    df = long_df.copy()
    if "split" not in df.columns:
        df = add_time_block_split(df)
    if any(column in df.columns for column in ("next_observed_value", "time_to_next_seen")):
        # These columns may exist in canonical data, but DwellNet must not consume them as features.
        pass
    table = build_dwell_token_table(df, target_mode=target_mode)
    dataset_id = str(table["dataset_id"].iloc[0])
    source_type = str(table["source_type"].iloc[0])
    spec = dataset_spec(dataset_id)
    branch_ids = [int(value) for value in sorted(df["branch_id"].dropna().unique().tolist())]
    branch_names, load_types = _branch_metadata(df, branch_ids)

    target = table[[f"target_b{branch_id:02d}" for branch_id in branch_ids]].to_numpy(dtype=float)
    observed = table[[f"observed_b{branch_id:02d}" for branch_id in branch_ids]].to_numpy(dtype=float)
    observed_mask = table[[f"observed_mask_b{branch_id:02d}" for branch_id in branch_ids]].to_numpy(dtype=bool)
    last_observed, last_available, time_since = _post_dwell_last_observed(observed)
    split = table["split"].astype(str).to_numpy()
    train_mask = split == "train"

    target_normalizer = DwellNormalizer.fit(
        target,
        train_mask=train_mask,
        branch_ids=branch_ids,
        mode=normalization_mode,
        value_kind="target",
    )
    residual = target - np.where(np.isfinite(last_observed), last_observed, 0.0)
    residual_normalizer = DwellNormalizer.fit(
        residual,
        train_mask=train_mask,
        branch_ids=branch_ids,
        mode=normalization_mode,
        value_kind="residual",
    )
    output_normalizer = target_normalizer if output_mode == "absolute" else residual_normalizer
    output_target = target if output_mode == "absolute" else residual

    target_scaled = target_normalizer.transform(target)
    output_target_scaled = output_normalizer.transform(output_target)
    observed_scaled = target_normalizer.transform(observed, fill_value=0.0)
    last_scaled = target_normalizer.transform(last_observed, fill_value=0.0)

    finite_since = time_since[train_mask]
    finite_since = finite_since[np.isfinite(finite_since)]
    since_denom = float(np.nanmax(finite_since)) if len(finite_since) else 1.0
    if not np.isfinite(since_denom) or since_denom <= 0:
        since_denom = 1.0
    time_since_norm = np.where(np.isfinite(time_since), time_since / since_denom, 1.0)
    time_since_norm = np.clip(time_since_norm, 0.0, 5.0).astype(np.float32)

    blocks: list[np.ndarray] = [
        observed_scaled[:, :, None],
        observed_mask.astype(np.float32)[:, :, None],
        last_scaled[:, :, None],
        last_available.astype(np.float32)[:, :, None],
        time_since_norm[:, :, None],
        observed_mask.astype(np.float32)[:, :, None],
    ]
    feature_names = [
        "observed_dwell_value_scaled",
        "observed_mask",
        "last_observed_dwell_value_scaled",
        "last_observed_available",
        "time_since_last_seen_norm",
        "selected_branch_mask",
    ]

    token_count = len(table)
    static_features, static_names = _static_branch_features(
        branch_ids,
        load_types,
        token_count,
        include_branch_static=include_branch_static,
    )
    blocks.append(static_features)
    feature_names.extend(static_names)

    if include_time_features:
        dwell_id = table["dwell_id"].to_numpy(dtype=float)
        scan_cycle_id = table["scan_cycle_id"].to_numpy(dtype=float)
        scalar_features = {
            "time_s_norm": _safe_normalize(table["time_s"].to_numpy(dtype=float), train_mask),
            "time_index_norm": _safe_normalize(table["time_index_end"].to_numpy(dtype=float), train_mask),
            "dwell_id_norm": _safe_normalize(dwell_id, train_mask),
            "scan_cycle_id_norm": _safe_normalize(scan_cycle_id, train_mask),
            "cycle_position_norm": (table["cycle_position"].to_numpy(dtype=float) / max(1, len(branch_ids) - 1)).astype(
                np.float32
            ),
            "scan_cycle_mod_8_norm": ((scan_cycle_id % 8) / 7.0).astype(np.float32),
            "dwell_id_mod_16_norm": ((dwell_id % 16) / 15.0).astype(np.float32),
        }
        for name, values in scalar_features.items():
            blocks.append(np.repeat(values.reshape(-1, 1, 1), len(branch_ids), axis=1))
            feature_names.append(name)

    features = np.concatenate(blocks, axis=2).astype(np.float32)
    forbidden = [name for name in feature_names if any(token in name for token in FORBIDDEN_DWELL_FEATURE_TOKENS)]
    if forbidden:
        raise ValueError(f"DwellNet feature leakage detected: {forbidden}")

    observed_values_physical = np.where(np.isfinite(observed), observed, np.nan).astype(np.float32)
    return DwellSeries(
        dataset_id=dataset_id,
        source_type=source_type,
        target_mode=target_mode,
        reconstruction_mode=reconstruction_mode,
        output_mode=output_mode,
        normalization_mode=normalization_mode,
        include_time_features=include_time_features,
        include_branch_static=include_branch_static,
        branch_ids=branch_ids,
        branch_names=branch_names,
        load_types=load_types,
        dwell_id=table["dwell_id"].to_numpy(dtype=np.int64),
        scan_cycle_id=table["scan_cycle_id"].to_numpy(dtype=np.int64),
        cycle_position=table["cycle_position"].to_numpy(dtype=np.int64),
        selected_branch=table["selected_branch"].to_numpy(dtype=np.int64),
        split=split,
        time_s=table["time_s"].to_numpy(dtype=float),
        time_index_start=table["time_index_start"].to_numpy(dtype=np.int64),
        time_index_end=table["time_index_end"].to_numpy(dtype=np.int64),
        dt_s=table["dt_s"].to_numpy(dtype=float),
        dwell_s_effective=table["dwell_s_effective"].to_numpy(dtype=float),
        target_physical=target.astype(np.float32),
        target_scaled=target_scaled.astype(np.float32),
        output_target_scaled=output_target_scaled.astype(np.float32),
        residual_physical=residual.astype(np.float32),
        observed_values_physical=observed_values_physical,
        observed_mask=observed_mask.astype(np.float32),
        last_observed_physical=np.where(np.isfinite(last_observed), last_observed, np.nan).astype(np.float32),
        last_observed_scaled=last_scaled.astype(np.float32),
        last_observed_available=last_available.astype(np.float32),
        time_since_last_seen=time_since.astype(np.float32),
        features=features,
        feature_names=feature_names,
        target_normalizer=target_normalizer,
        output_normalizer=output_normalizer,
        token_table=table,
        long=df,
    )


def validate_window_cycles(series: DwellSeries, window_cycles: int) -> int:
    spec = dataset_spec(series.dataset_id)
    supported = tuple(int(value) for value in spec["window_cycles"])
    if int(window_cycles) not in supported:
        raise ValueError(f"{series.dataset_id} supports window_cycles {supported}, got {window_cycles}")
    return int(window_cycles) * series.branch_count


def dwell_window_end_positions(
    series: DwellSeries,
    *,
    split: str,
    window_cycles: int,
    max_windows: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    window_tokens = validate_window_cycles(series, window_cycles)
    positions = np.flatnonzero(series.split == split)
    positions = positions[positions >= window_tokens - 1]
    if len(positions) == 0:
        raise ValueError(f"No {split} dwell windows available for window_cycles={window_cycles}")
    if max_windows is not None and len(positions) > max_windows:
        rng = np.random.default_rng(seed)
        positions = np.sort(rng.choice(positions, size=max_windows, replace=False))
    return positions.astype(np.int64)


class DwellWindowDataset:
    def __init__(
        self,
        series: DwellSeries,
        *,
        split: str,
        window_cycles: int,
        max_windows: int | None = None,
        seed: int = 42,
    ):
        self.series = series
        self.window_cycles = int(window_cycles)
        self.window_tokens = validate_window_cycles(series, window_cycles)
        self.end_positions = dwell_window_end_positions(
            series,
            split=split,
            window_cycles=window_cycles,
            max_windows=max_windows,
            seed=seed,
        )
        self.split = split
        self._torch = require_torch()

    def __len__(self) -> int:
        return int(len(self.end_positions))

    def __getitem__(self, index: int) -> dict[str, object]:
        end = int(self.end_positions[index])
        start = end - self.window_tokens + 1
        torch = self._torch
        return {
            "x": torch.as_tensor(self.series.features[start : end + 1], dtype=torch.float32),
            "target": torch.as_tensor(self.series.output_target_scaled[end], dtype=torch.float32),
            "target_physical": torch.as_tensor(self.series.target_physical[end], dtype=torch.float32),
            "base_physical": torch.as_tensor(
                np.where(np.isfinite(self.series.last_observed_physical[end]), self.series.last_observed_physical[end], 0.0),
                dtype=torch.float32,
            ),
            "observed_values": torch.as_tensor(
                np.where(np.isfinite(self.series.observed_values_physical[end]), self.series.observed_values_physical[end], 0.0),
                dtype=torch.float32,
            ),
            "observed_mask": torch.as_tensor(self.series.observed_mask[end], dtype=torch.float32),
            "dwell_s_effective": torch.as_tensor(self.series.dwell_s_effective[end], dtype=torch.float32),
            "end_position": torch.as_tensor(end, dtype=torch.long),
        }


def normalization_metadata(series: DwellSeries) -> dict[str, Any]:
    return {
        "dataset_id": series.dataset_id,
        "target_mode": series.target_mode,
        "output_mode": series.output_mode,
        "normalization_mode": series.normalization_mode,
        "target_normalizer": series.target_normalizer.to_dict(),
        "output_normalizer": series.output_normalizer.to_dict(),
        "feature_names": series.feature_names,
        "branch_ids": series.branch_ids,
        "branch_names": series.branch_names,
        "load_types": series.load_types,
        "token_count": int(len(series.dwell_id)),
        "include_time_features": series.include_time_features,
        "include_branch_static": series.include_branch_static,
        "dataset_spec": dataset_spec(series.dataset_id),
    }
