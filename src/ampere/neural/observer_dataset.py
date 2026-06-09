from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ampere.neural.dwell_dataset import (
    DwellSeries,
    build_dwell_series,
    dataset_spec,
    dwell_window_end_positions,
    normalization_metadata,
    validate_window_cycles,
)
from ampere.neural.utils import require_torch


OBSERVER_MODEL_TYPES = (
    "observer_gru",
    "observer_prior_only",
    "observer_fixed_selected_gain",
    "observer_mlp",
    "observer_mlp_prior_only",
    "observer_mlp_fixed_selected_gain",
    "observer_mlp_learned_selected_gain",
    "observer_mlp_sparse_offbranch_gain",
)
OBSERVER_LOSS_VARIANTS = (
    "observer_supervised",
    "observer_transition",
    "observer_energy",
    "observer_gain_regularized",
)
FORBIDDEN_OBSERVER_FEATURE_TOKENS = ("next_observed", "time_to_next", "root_power")


@dataclass(frozen=True)
class ObserverSeries:
    dwell: DwellSeries
    pre_observed_physical: np.ndarray
    pre_observed_scaled: np.ndarray
    pre_observed_available: np.ndarray
    pre_time_since_last_seen: np.ndarray
    pre_time_since_last_seen_norm: np.ndarray
    transition_mask: np.ndarray
    transition_thresholds: np.ndarray

    @property
    def dataset_id(self) -> str:
        return self.dwell.dataset_id

    @property
    def target_mode(self) -> str:
        return self.dwell.target_mode

    @property
    def reconstruction_mode(self) -> str:
        return self.dwell.reconstruction_mode

    @property
    def branch_count(self) -> int:
        return self.dwell.branch_count

    @property
    def feature_dim(self) -> int:
        return self.dwell.feature_dim


def _pre_dwell_last_observed(observed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    token_count, branch_count = observed.shape
    last_values = np.full((token_count, branch_count), np.nan, dtype=float)
    last_available = np.zeros((token_count, branch_count), dtype=bool)
    time_since = np.full((token_count, branch_count), np.nan, dtype=float)
    current = np.full(branch_count, np.nan, dtype=float)
    seen_at = np.full(branch_count, -1, dtype=int)

    for token_idx in range(token_count):
        last_values[token_idx] = current
        available = seen_at >= 0
        last_available[token_idx] = available
        time_since[token_idx, available] = token_idx - seen_at[available]

        observed_now = np.isfinite(observed[token_idx])
        current[observed_now] = observed[token_idx, observed_now]
        seen_at[observed_now] = token_idx

    return last_values, last_available, time_since


def _normalize_time_since(time_since: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    finite = time_since[train_mask]
    finite = finite[np.isfinite(finite)]
    denom = float(np.nanmax(finite)) if len(finite) else 1.0
    if not np.isfinite(denom) or denom <= 0:
        denom = 1.0
    values = np.where(np.isfinite(time_since), time_since / denom, 1.0)
    return np.clip(values, 0.0, 5.0).astype(np.float32)


def _transition_mask(target: np.ndarray, split: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = target[split == "train"]
    if len(train) > 1:
        thresholds = np.nanpercentile(np.abs(np.diff(train, axis=0)), 75, axis=0)
    else:
        thresholds = np.zeros(target.shape[1], dtype=float)
    thresholds = np.where(np.isfinite(thresholds), thresholds, 0.0)
    diffs = np.vstack([np.zeros((1, target.shape[1]), dtype=float), np.abs(np.diff(target, axis=0))])
    return (diffs > thresholds.reshape(1, -1)).astype(np.float32), thresholds.astype(np.float32)


def build_observer_series(
    long_df: pd.DataFrame,
    *,
    target_mode: str = "dwell_mean_power",
    reconstruction_mode: str = "online_safe",
    output_mode: str = "residual_dwell",
    normalization_mode: str = "branchwise",
    include_time_features: bool = True,
    include_branch_static: bool = True,
) -> ObserverSeries:
    if output_mode != "residual_dwell":
        raise ValueError("Base DwellObserver currently uses residual_dwell semantics only")
    dwell = build_dwell_series(
        long_df,
        target_mode=target_mode,
        reconstruction_mode=reconstruction_mode,
        output_mode=output_mode,
        normalization_mode=normalization_mode,
        include_time_features=include_time_features,
        include_branch_static=include_branch_static,
    )
    forbidden = [name for name in dwell.feature_names if any(token in name for token in FORBIDDEN_OBSERVER_FEATURE_TOKENS)]
    if forbidden:
        raise ValueError(f"DwellObserver feature leakage detected: {forbidden}")

    pre_values, pre_available, pre_time_since = _pre_dwell_last_observed(dwell.observed_values_physical)
    pre_scaled = dwell.target_normalizer.transform(pre_values, fill_value=0.0)
    pre_time_since_norm = _normalize_time_since(pre_time_since, dwell.split == "train")
    transitions, transition_thresholds = _transition_mask(dwell.target_physical, dwell.split)
    return ObserverSeries(
        dwell=dwell,
        pre_observed_physical=np.where(np.isfinite(pre_values), pre_values, np.nan).astype(np.float32),
        pre_observed_scaled=pre_scaled.astype(np.float32),
        pre_observed_available=pre_available.astype(np.float32),
        pre_time_since_last_seen=pre_time_since.astype(np.float32),
        pre_time_since_last_seen_norm=pre_time_since_norm.astype(np.float32),
        transition_mask=transitions.astype(np.float32),
        transition_thresholds=transition_thresholds.astype(np.float32),
    )


class ObserverWindowDataset:
    def __init__(
        self,
        series: ObserverSeries,
        *,
        split: str,
        window_cycles: int,
        max_windows: int | None = None,
        seed: int = 42,
    ):
        self.series = series
        self.window_cycles = int(window_cycles)
        self.window_tokens = validate_window_cycles(series.dwell, window_cycles)
        self.end_positions = dwell_window_end_positions(
            series.dwell,
            split=split,
            window_cycles=window_cycles,
            max_windows=max_windows,
            seed=seed,
        )
        self.split = split
        self._torch = require_torch()
        self._feature_index = {name: idx for idx, name in enumerate(series.dwell.feature_names)}

    def __len__(self) -> int:
        return int(len(self.end_positions))

    def _prior_context(self, start: int, end: int) -> np.ndarray:
        x = np.array(self.series.dwell.features[start : end + 1], copy=True)
        final = -1
        indices = self._feature_index
        if "observed_dwell_value_scaled" in indices:
            x[final, :, indices["observed_dwell_value_scaled"]] = 0.0
        if "last_observed_dwell_value_scaled" in indices:
            x[final, :, indices["last_observed_dwell_value_scaled"]] = self.series.pre_observed_scaled[end]
        if "last_observed_available" in indices:
            x[final, :, indices["last_observed_available"]] = self.series.pre_observed_available[end]
        if "time_since_last_seen_norm" in indices:
            x[final, :, indices["time_since_last_seen_norm"]] = self.series.pre_time_since_last_seen_norm[end]
        return x.astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, object]:
        end = int(self.end_positions[index])
        start = end - self.window_tokens + 1
        torch = self._torch
        observed_mask = self.series.dwell.observed_mask[end].astype(np.float32)
        selected_branch_index = int(np.argmax(observed_mask))
        observed_values = np.where(
            np.isfinite(self.series.dwell.observed_values_physical[end]),
            self.series.dwell.observed_values_physical[end],
            0.0,
        ).astype(np.float32)
        y_obs_physical = float(observed_values[selected_branch_index])
        y_obs_scaled = float(self.series.dwell.target_normalizer.transform(observed_values.reshape(1, -1))[0, selected_branch_index])
        pre_base_physical = np.where(
            np.isfinite(self.series.pre_observed_physical[end]),
            self.series.pre_observed_physical[end],
            self.series.dwell.target_normalizer.mean,
        ).astype(np.float32)
        pre_base_scaled = self.series.pre_observed_scaled[end].astype(np.float32)
        return {
            "x": torch.as_tensor(self._prior_context(start, end), dtype=torch.float32),
            "target_scaled": torch.as_tensor(self.series.dwell.target_scaled[end], dtype=torch.float32),
            "target_physical": torch.as_tensor(self.series.dwell.target_physical[end], dtype=torch.float32),
            "pre_base_scaled": torch.as_tensor(pre_base_scaled, dtype=torch.float32),
            "pre_base_physical": torch.as_tensor(pre_base_physical, dtype=torch.float32),
            "pre_time_since_last_seen_norm": torch.as_tensor(
                self.series.pre_time_since_last_seen_norm[end],
                dtype=torch.float32,
            ),
            "observed_values": torch.as_tensor(observed_values, dtype=torch.float32),
            "observed_mask": torch.as_tensor(observed_mask, dtype=torch.float32),
            "selected_branch_index": torch.as_tensor(selected_branch_index, dtype=torch.long),
            "y_obs_physical": torch.as_tensor(y_obs_physical, dtype=torch.float32),
            "y_obs_scaled": torch.as_tensor(y_obs_scaled, dtype=torch.float32),
            "transition_mask": torch.as_tensor(self.series.transition_mask[end], dtype=torch.float32),
            "dwell_s_effective": torch.as_tensor(self.series.dwell.dwell_s_effective[end], dtype=torch.float32),
            "end_position": torch.as_tensor(end, dtype=torch.long),
        }


def observer_normalization_metadata(series: ObserverSeries) -> dict[str, Any]:
    metadata = normalization_metadata(series.dwell)
    metadata["observer_pre_dwell_context"] = True
    metadata["observer_dataset_spec"] = dataset_spec(series.dataset_id)
    metadata["transition_thresholds_train_only"] = series.transition_thresholds.tolist()
    return metadata
