from __future__ import annotations

import pandas as pd

from ampere.evaluation.leaderboard import rank_leaderboard
from ampere.evaluation.metrics import MetricTables, evaluate_predictions


def evaluate_classical_predictions(predictions: pd.DataFrame) -> MetricTables:
    return evaluate_predictions(predictions)


def build_ranked_leaderboard(predictions: pd.DataFrame) -> pd.DataFrame:
    tables = evaluate_predictions(predictions)
    return rank_leaderboard(tables.overall)
