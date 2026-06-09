from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.60
    val_fraction: float = 0.20
    test_fraction: float = 0.20
    time_col: str = "time_index"

    def validate(self) -> None:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Split fractions must sum to 1.0, got {total}")
        if min(self.train_fraction, self.val_fraction, self.test_fraction) <= 0:
            raise ValueError("Split fractions must be positive")


def add_time_block_split(df: pd.DataFrame, config: SplitConfig | None = None) -> pd.DataFrame:
    cfg = config or SplitConfig()
    cfg.validate()
    if cfg.time_col not in df.columns:
        raise ValueError(f"Missing split time column: {cfg.time_col}")

    result = df.copy()
    unique_times = pd.Index(sorted(result[cfg.time_col].dropna().unique()))
    if len(unique_times) < 3:
        raise ValueError("Need at least three unique timestamps for train/val/test split")

    train_cut = int(len(unique_times) * cfg.train_fraction)
    val_cut = int(len(unique_times) * (cfg.train_fraction + cfg.val_fraction))
    train_cut = max(1, min(train_cut, len(unique_times) - 2))
    val_cut = max(train_cut + 1, min(val_cut, len(unique_times) - 1))

    rank_map = pd.Series(range(len(unique_times)), index=unique_times)
    ranks = result[cfg.time_col].map(rank_map).to_numpy()
    split = pd.Series("test", index=result.index, dtype="object")
    split.iloc[ranks < train_cut] = "train"
    split.iloc[(ranks >= train_cut) & (ranks < val_cut)] = "val"
    result["split"] = split.to_numpy()
    return result


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "split" not in df.columns:
        raise ValueError("Dataframe has no split column")
    return (
        df.groupby("split", sort=False)
        .agg(rows=("split", "size"), min_time_index=("time_index", "min"), max_time_index=("time_index", "max"))
        .reset_index()
    )


def assert_time_block_no_overlap(df: pd.DataFrame) -> None:
    if "split" not in df.columns:
        raise AssertionError("Dataframe has no split column")
    split_times = {
        split: set(group["time_index"].unique().tolist())
        for split, group in df.groupby("split", sort=False)
    }
    names = list(split_times)
    for idx, left in enumerate(names):
        for right in names[idx + 1 :]:
            overlap = split_times[left] & split_times[right]
            if overlap:
                raise AssertionError(f"Time-block split overlap between {left} and {right}: {len(overlap)}")
