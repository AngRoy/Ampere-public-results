from __future__ import annotations

import pandas as pd


def rank_leaderboard(overall_metrics: pd.DataFrame) -> pd.DataFrame:
    rank_group_cols = ["dataset_id"]
    rank_group_cols.extend(
        [column for column in ["reconstruction_mode", "target_mode"] if column in overall_metrics.columns]
    )
    sort_cols = rank_group_cols + ["mae", "weighted_energy_error", "rmse"]
    ranked = overall_metrics.sort_values(sort_cols, kind="mergesort").copy()
    ranked["rank"] = ranked.groupby(rank_group_cols).cumcount() + 1
    first_cols = rank_group_cols + ["rank", "method"]
    return ranked[first_cols + [col for col in ranked.columns if col not in first_cols]]
