from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ampere.data.canonical import write_dataframe_prefer_parquet
from ampere.evaluation.leaderboard import rank_leaderboard
from ampere.evaluation.metrics import evaluate_predictions
from ampere.features.splits import add_time_block_split, assert_time_block_no_overlap, split_summary
from ampere.models.tree_baselines import MODEL_LEVELS, TREE_MODEL_FAMILIES, run_tree_baseline
from ampere.utils.paths import as_repo_relative


DATASET_LONG_INPUTS = {
    "rlc_sample": "rlc_sample_long.parquet",
    "appliance_8ch": "appliance_8ch_long.parquet",
}

CLASSICAL_PREDICTION_INPUTS = {
    "rlc_sample": "rlc_sample_classical_predictions.parquet",
    "appliance_8ch": "appliance_8ch_classical_predictions.parquet",
}

SERIOUS_CLASSICAL_EXCLUDE = {"mean_per_branch"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AMPERE RF / boosting tree baselines.")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_LONG_INPUTS), default=sorted(DATASET_LONG_INPUTS))
    parser.add_argument("--modes", nargs="+", choices=["online_safe", "offline"], default=["online_safe", "offline"])
    parser.add_argument(
        "--target-modes",
        nargs="+",
        choices=["raw_signed_power", "dwell_mean_power"],
        default=["raw_signed_power", "dwell_mean_power"],
    )
    parser.add_argument("--model-families", nargs="+", choices=list(TREE_MODEL_FAMILIES), default=list(TREE_MODEL_FAMILIES))
    parser.add_argument("--model-levels", nargs="+", choices=list(MODEL_LEVELS), default=list(MODEL_LEVELS))
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "outputs" / "data" / "processed")
    parser.add_argument("--classical-dir", type=Path, default=ROOT / "outputs" / "reconstruction" / "classical")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "reconstruction" / "tree")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "outputs" / "figures" / "tree")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "tree_baseline_report.md")
    parser.add_argument("--predict-splits", nargs="+", choices=["train", "val", "test"], default=["test"])
    parser.add_argument("--max-train-rows", type=int, default=60_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def format_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if abs(value) >= 1000 or (0 < abs(value) < 0.001):
            return f"{value:.4e}"
        return f"{value:.4f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> list[str]:
    if df.empty:
        return ["_No rows._"]
    table = df[columns].copy()
    if max_rows is not None:
        table = table.head(max_rows)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in table.itertuples(index=False):
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return lines


def load_canonical_long(dataset_id: str, processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / DATASET_LONG_INPUTS[dataset_id]
    df = pd.read_parquet(path)
    df = add_time_block_split(df)
    assert_time_block_no_overlap(df)
    return df


def prepare_classical_predictions_for_target(
    *,
    dataset_id: str,
    target_mode: str,
    canonical: pd.DataFrame,
    classical_dir: Path,
) -> pd.DataFrame:
    path = classical_dir / CLASSICAL_PREDICTION_INPUTS[dataset_id]
    predictions = pd.read_parquet(path)
    predictions = add_time_block_split(predictions)
    predictions = predictions[predictions["split"].eq("test")].copy()

    if target_mode == "dwell_mean_power":
        target_map = canonical[["time_index", "branch_id", "P_dwell_mean"]].drop_duplicates(
            ["time_index", "branch_id"]
        )
        predictions = predictions.drop(columns=["P_true"], errors="ignore").merge(
            target_map,
            on=["time_index", "branch_id"],
            how="left",
        )
        predictions = predictions.rename(columns={"P_dwell_mean": "P_true"})
        predictions["power_representation"] = "dwell_mean_power"
    else:
        predictions["power_representation"] = "raw_signed_power"

    predictions["reconstruction_mode"] = "classical"
    predictions["target_mode"] = target_mode
    predictions["model_family"] = "classical"
    predictions["model_level"] = "row_interpolation"
    return predictions


def evaluate_ranked(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = evaluate_predictions(predictions)
    return rank_leaderboard(tables.overall), tables.branch


def best_rows_for_comparison(tree_leaderboard: pd.DataFrame, classical_leaderboard: pd.DataFrame) -> pd.DataFrame:
    rows = []
    datasets = sorted(set(tree_leaderboard["dataset_id"]).union(classical_leaderboard["dataset_id"]))
    target_modes = sorted(set(tree_leaderboard["target_mode"]).union(classical_leaderboard["target_mode"]))
    for dataset_id in datasets:
        for target_mode in target_modes:
            classical = classical_leaderboard[
                (classical_leaderboard["dataset_id"] == dataset_id)
                & (classical_leaderboard["target_mode"] == target_mode)
                & (~classical_leaderboard["method"].isin(SERIOUS_CLASSICAL_EXCLUDE))
            ].sort_values("mae")
            online = tree_leaderboard[
                (tree_leaderboard["dataset_id"] == dataset_id)
                & (tree_leaderboard["target_mode"] == target_mode)
                & (tree_leaderboard["reconstruction_mode"] == "online_safe")
            ].sort_values("mae")
            offline = tree_leaderboard[
                (tree_leaderboard["dataset_id"] == dataset_id)
                & (tree_leaderboard["target_mode"] == target_mode)
                & (tree_leaderboard["reconstruction_mode"] == "offline")
            ].sort_values("mae")
            if classical.empty or online.empty or offline.empty:
                continue
            c = classical.iloc[0]
            o = online.iloc[0]
            f = offline.iloc[0]
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "target_mode": target_mode,
                    "best_serious_classical": c["method"],
                    "classical_mae": c["mae"],
                    "classical_dwell_mae": c["dwell_mae"],
                    "classical_weighted_energy_error": c["weighted_energy_error"],
                    "best_online_tree": o["method"],
                    "online_model_family": o["model_family"],
                    "online_model_level": o["model_level"],
                    "online_mae": o["mae"],
                    "online_dwell_mae": o["dwell_mae"],
                    "online_weighted_energy_error": o["weighted_energy_error"],
                    "online_improves_mae": bool(o["mae"] < c["mae"]),
                    "online_improves_dwell_mae": bool(o["dwell_mae"] < c["dwell_mae"]),
                    "best_offline_tree": f["method"],
                    "offline_model_family": f["model_family"],
                    "offline_model_level": f["model_level"],
                    "offline_mae": f["mae"],
                    "offline_dwell_mae": f["dwell_mae"],
                    "offline_weighted_energy_error": f["weighted_energy_error"],
                    "offline_improves_mae": bool(f["mae"] < c["mae"]),
                    "offline_improves_dwell_mae": bool(f["dwell_mae"] < c["dwell_mae"]),
                }
            )
    return pd.DataFrame(rows)


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def generate_tree_figures(
    *,
    tree_predictions: pd.DataFrame,
    classical_predictions: pd.DataFrame,
    tree_leaderboard: pd.DataFrame,
    classical_leaderboard: pd.DataFrame,
    tree_branch_metrics: pd.DataFrame,
    classical_branch_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    feature_importance: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    plt = _load_matplotlib()
    if plt is None:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    for row in comparison.itertuples(index=False):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        labels = ["classical", "online tree", "offline tree"]
        values = [row.classical_mae, row.online_mae, row.offline_mae]
        ax.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b"])
        ax.set_title(f"{row.dataset_id} {row.target_mode}: MAE comparison")
        ax.set_ylabel("MAE")
        ax.grid(axis="y", alpha=0.25)
        path = output_dir / f"{row.dataset_id}_{row.target_mode}_mae_comparison.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(str(path))

    for dataset_id in sorted(tree_predictions["dataset_id"].unique()):
        target_mode = "raw_signed_power"
        rows = comparison[(comparison["dataset_id"] == dataset_id) & (comparison["target_mode"] == target_mode)]
        if rows.empty:
            continue
        row = rows.iloc[0]
        tree_branch = tree_branch_metrics[
            (tree_branch_metrics["dataset_id"] == dataset_id)
            & (tree_branch_metrics["target_mode"] == target_mode)
            & (tree_branch_metrics["reconstruction_mode"] == "online_safe")
            & (tree_branch_metrics["method"] == row["best_online_tree"])
            & (tree_branch_metrics["model_family"] == row["online_model_family"])
            & (tree_branch_metrics["model_level"] == row["online_model_level"])
        ].sort_values("branch_id")
        classical_branch = classical_branch_metrics[
            (classical_branch_metrics["dataset_id"] == dataset_id)
            & (classical_branch_metrics["target_mode"] == target_mode)
            & (classical_branch_metrics["method"] == row["best_serious_classical"])
        ].sort_values("branch_id")
        if not tree_branch.empty and not classical_branch.empty:
            merged = tree_branch[["branch_id", "abs_energy_error_Wh"]].merge(
                classical_branch[["branch_id", "abs_energy_error_Wh"]],
                on="branch_id",
                suffixes=("_tree", "_classical"),
            )
            fig, ax = plt.subplots(figsize=(8, 4.5))
            x = range(len(merged))
            ax.bar([v - 0.18 for v in x], merged["abs_energy_error_Wh_classical"], width=0.36, label="classical")
            ax.bar([v + 0.18 for v in x], merged["abs_energy_error_Wh_tree"], width=0.36, label="online tree")
            ax.set_xticks(list(x), [str(int(v)) for v in merged["branch_id"]])
            ax.set_title(f"{dataset_id}: per-branch energy error")
            ax.set_xlabel("branch_id")
            ax.set_ylabel("abs signed-energy error, Wh")
            ax.legend()
            ax.grid(axis="y", alpha=0.25)
            path = output_dir / f"{dataset_id}_raw_per_branch_energy_error.png"
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            generated.append(str(path))

        tree_best = tree_predictions[
            (tree_predictions["dataset_id"] == dataset_id)
            & (tree_predictions["target_mode"] == target_mode)
            & (tree_predictions["reconstruction_mode"] == "online_safe")
            & (tree_predictions["method"] == row["best_online_tree"])
            & (tree_predictions["model_family"] == row["online_model_family"])
            & (tree_predictions["model_level"] == row["online_model_level"])
        ]
        classical_best = classical_predictions[
            (classical_predictions["dataset_id"] == dataset_id)
            & (classical_predictions["target_mode"] == target_mode)
            & (classical_predictions["method"] == row["best_serious_classical"])
        ]
        if not tree_best.empty and not classical_best.empty:
            branch_id = int(sorted(tree_best["branch_id"].unique())[0])
            tree_branch_pred = tree_best[tree_best["branch_id"] == branch_id].sort_values("time_index")
            classical_branch_pred = classical_best[classical_best["branch_id"] == branch_id].sort_values("time_index")
            start = float(tree_branch_pred["time_s"].min())
            stop = start + (30.0 if dataset_id == "rlc_sample" else 180.0)
            tree_branch_pred = tree_branch_pred[tree_branch_pred["time_s"] <= stop]
            classical_branch_pred = classical_branch_pred[classical_branch_pred["time_s"] <= stop]
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.plot(tree_branch_pred["time_s"], tree_branch_pred["P_true"], color="black", label="P_true", linewidth=1.3)
            ax.plot(tree_branch_pred["time_s"], tree_branch_pred["P_hat"], label="online tree", linewidth=1.0)
            ax.plot(classical_branch_pred["time_s"], classical_branch_pred["P_hat"], label="serious classical", linewidth=1.0)
            observed = tree_branch_pred[tree_branch_pred["is_observed"].astype(bool)]
            ax.scatter(observed["time_s"], observed["P_observed"], color="#d62728", s=12, label="observed")
            ax.set_title(f"{dataset_id}: tree vs classical overlay, branch {branch_id}")
            ax.set_xlabel("time_s")
            ax.set_ylabel("power")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.25)
            path = output_dir / f"{dataset_id}_raw_overlay_tree_vs_classical.png"
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            generated.append(str(path))

    if not feature_importance.empty:
        for dataset_id in sorted(feature_importance["dataset_id"].unique()):
            subset = feature_importance[feature_importance["dataset_id"] == dataset_id].head(15)
            if subset.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 5.5))
            ax.barh(subset["feature"][::-1], subset["importance"][::-1], color="#4c78a8")
            ax.set_title(f"{dataset_id}: top feature importances")
            ax.set_xlabel("importance")
            path = output_dir / f"{dataset_id}_feature_importance.png"
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            generated.append(str(path))
    return generated


def write_report(
    *,
    report_path: Path,
    split_summaries: dict[str, pd.DataFrame],
    tree_leaderboard: pd.DataFrame,
    comparison: pd.DataFrame,
    outputs: dict[str, str],
    figures: list[str],
    metadata: list[dict[str, object]],
    classical_full_leaderboard: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tree Baseline Report",
        "",
        "Generated by `python scripts/run_tree_baselines.py --datasets rlc_sample appliance_8ch --modes online_safe offline --target-modes raw_signed_power dwell_mean_power`.",
        "",
        "## Scope",
        "",
        "This Stage 2 run trains RF / ExtraTrees / HistGradientBoosting tabular baselines only. No neural, physics-guided, adaptive-sampling, or RL models were implemented.",
        "",
        "Inputs are Phase 1 canonical outputs and Stage 1 classical outputs. Raw files under `Data/` were not modified.",
        "",
        "## Split Strategy",
        "",
        "Each dataset uses a time-block split by `time_index`: first 60% train, next 20% validation reserve, final 20% test. Stage 2 leaderboards are computed on the test block only.",
        "",
    ]
    for dataset_id, summary in split_summaries.items():
        lines.append(f"### {dataset_id}")
        lines.extend(markdown_table(summary, ["split", "rows", "min_time_index", "max_time_index"]))
        lines.append("")

    lines.extend(
        [
            "## Feature Policy",
            "",
            "- `online_safe` excludes `next_observed_value`, `time_to_next_seen`, and future-derived indicators.",
            "- `offline` may use `next_observed_value` and `time_to_next_seen`, comparable to interpolation-style post-processing.",
            "- Direct truth columns are excluded from features: `P_true`, `P_raw_signed`, `P_dwell_mean`, `P_clipped_nonnegative`, and `root_power`.",
            "- `P_observed` is used only as the sparse observed channel value, filled to zero when unavailable and paired with availability flags.",
            "- Default Stage 2 does not use synthesized `root_power` because it is an oracle aggregate in the current simulation outputs.",
            "",
            "## Models",
            "",
            "- `random_forest`: scikit-learn `RandomForestRegressor`, 24 trees, capped depth, sampled training rows.",
            "- `extra_trees`: scikit-learn `ExtraTreesRegressor`, 24 trees, capped depth, sampled training rows.",
            "- `hist_gradient_boosting`: scikit-learn `HistGradientBoostingRegressor`, 80 iterations.",
            "- Optional LightGBM/XGBoost are installed locally but intentionally not part of the default Stage 2 report to keep the project baseline dependency surface small.",
            "",
            "## Tree Leaderboard",
            "",
        ]
    )
    leaderboard_cols = [
        "dataset_id",
        "reconstruction_mode",
        "target_mode",
        "rank",
        "method",
        "model_family",
        "model_level",
        "mae",
        "rmse",
        "dwell_mae",
        "weighted_energy_error",
        "observed_point_mae",
        "unobserved_point_mae",
    ]
    lines.extend(markdown_table(tree_leaderboard.sort_values(["dataset_id", "target_mode", "reconstruction_mode", "rank"]), leaderboard_cols, max_rows=80))
    lines.extend(
        [
            "",
            "## Classical Comparison",
            "",
            "Serious classical ranking excludes `mean_per_branch`, which remains a sanity baseline only. Classical predictions are rescored on the same final 20% test block for fairness.",
            "",
        ]
    )
    comparison_cols = [
        "dataset_id",
        "target_mode",
        "best_serious_classical",
        "classical_mae",
        "best_online_tree",
        "online_model_level",
        "online_mae",
        "online_improves_mae",
        "best_offline_tree",
        "offline_model_level",
        "offline_mae",
        "offline_improves_mae",
        "classical_dwell_mae",
        "online_dwell_mae",
        "offline_dwell_mae",
    ]
    lines.extend(markdown_table(comparison, comparison_cols))
    lines.extend(["", "### Result-Specific Notes", ""])
    for row in comparison.itertuples(index=False):
        online_joint = row.online_improves_mae and row.online_improves_dwell_mae
        offline_joint = row.offline_improves_mae and row.offline_improves_dwell_mae
        online_energy = row.online_weighted_energy_error < row.classical_weighted_energy_error
        offline_energy = row.offline_weighted_energy_error < row.classical_weighted_energy_error
        lines.append(
            f"- `{row.dataset_id}` / `{row.target_mode}`: online tree joint MAE+dwell improvement = `{online_joint}`, "
            f"offline tree joint MAE+dwell improvement = `{offline_joint}`; "
            f"online signed-energy improvement = `{online_energy}`, offline signed-energy improvement = `{offline_energy}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Improvements are meaningful only when pointwise and dwell/window metrics improve together; signed energy alone can hide timing errors.",
            "- `dwell_mean_power` target rows repeat a dwell/window target at row level so the existing row-level metric code can compare models consistently.",
            "- For `raw_signed_power`, tree predictions are post-processed to exactly preserve observed selected-branch values. For `dwell_mean_power`, raw observations are not forced onto the dwell-mean target.",
            "",
            "## Loaded Stage 1 Full Leaderboard",
            "",
        ]
    )
    if not classical_full_leaderboard.empty:
        lines.extend(
            markdown_table(
                classical_full_leaderboard,
                ["dataset_id", "rank", "method", "mae", "dwell_mae", "weighted_energy_error"],
                max_rows=20,
            )
        )
    else:
        lines.append("_Stage 1 full leaderboard was not available._")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
        ]
    )
    for label, path in outputs.items():
        lines.append(f"- `{label}`: `{path}`")
    for path in figures:
        lines.append(f"- figure: `{as_repo_relative(path, ROOT)}`")
    lines.extend(
        [
            "",
            "## Run Metadata",
            "",
            "```json",
            json.dumps(metadata, indent=2),
            "```",
            "",
            "## Recommended Next Phase",
            "",
            "Review whether tree baselines beat serious classical baselines on pointwise and dwell/window metrics. Stage 3 physics-guided reconstruction should start only if the accepted Stage 2 comparison identifies gaps that constraints can plausibly address.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    predict_splits = set(args.predict_splits)
    all_predictions: list[pd.DataFrame] = []
    all_importance: list[pd.DataFrame] = []
    metadata: list[dict[str, object]] = []
    split_summaries: dict[str, pd.DataFrame] = {}
    canonical_by_dataset: dict[str, pd.DataFrame] = {}

    for dataset_id in args.datasets:
        canonical = load_canonical_long(dataset_id, args.processed_dir)
        canonical_by_dataset[dataset_id] = canonical
        split_summaries[dataset_id] = split_summary(canonical)
        for target_mode in args.target_modes:
            for reconstruction_mode in args.modes:
                for model_level in args.model_levels:
                    for model_family in args.model_families:
                        result = run_tree_baseline(
                            canonical,
                            dataset_id=dataset_id,
                            model_family=model_family,
                            model_level=model_level,
                            reconstruction_mode=reconstruction_mode,
                            target_mode=target_mode,
                            predict_splits=predict_splits,
                            max_train_rows=args.max_train_rows,
                            random_state=args.random_state,
                        )
                        all_predictions.append(result.predictions)
                        if not result.feature_importance.empty:
                            all_importance.append(result.feature_importance)
                        metadata.append(result.metadata)

    tree_predictions = pd.concat(all_predictions, ignore_index=True)
    tree_metrics = evaluate_predictions(tree_predictions)
    tree_leaderboard = rank_leaderboard(tree_metrics.overall)
    tree_branch_metrics = tree_metrics.branch
    feature_importance = pd.concat(all_importance, ignore_index=True) if all_importance else pd.DataFrame()

    classical_predictions = []
    for dataset_id in args.datasets:
        for target_mode in args.target_modes:
            classical_predictions.append(
                prepare_classical_predictions_for_target(
                    dataset_id=dataset_id,
                    target_mode=target_mode,
                    canonical=canonical_by_dataset[dataset_id],
                    classical_dir=args.classical_dir,
                )
            )
    classical_predictions_df = pd.concat(classical_predictions, ignore_index=True)
    classical_leaderboard, classical_branch_metrics = evaluate_ranked(classical_predictions_df)
    comparison = best_rows_for_comparison(tree_leaderboard, classical_leaderboard)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_meta = write_dataframe_prefer_parquet(tree_predictions, args.output_dir / "tree_predictions")
    leaderboard_csv = args.output_dir / "tree_leaderboard.csv"
    leaderboard_json = args.output_dir / "tree_leaderboard.json"
    branch_csv = args.output_dir / "tree_branch_metrics.csv"
    importance_csv = args.output_dir / "tree_feature_importance.csv"
    comparison_csv = args.output_dir / "tree_vs_classical_comparison.csv"
    classical_test_csv = args.output_dir / "classical_test_leaderboard.csv"

    tree_leaderboard.to_csv(leaderboard_csv, index=False)
    tree_leaderboard.to_json(leaderboard_json, orient="records", indent=2)
    tree_branch_metrics.to_csv(branch_csv, index=False)
    feature_importance.to_csv(importance_csv, index=False)
    comparison.to_csv(comparison_csv, index=False)
    classical_leaderboard.to_csv(classical_test_csv, index=False)

    classical_full_leaderboard_path = args.classical_dir / "classical_leaderboard.csv"
    classical_full_leaderboard = (
        pd.read_csv(classical_full_leaderboard_path) if classical_full_leaderboard_path.exists() else pd.DataFrame()
    )

    figures: list[str] = []
    if not args.skip_plots:
        figures = generate_tree_figures(
            tree_predictions=tree_predictions,
            classical_predictions=classical_predictions_df,
            tree_leaderboard=tree_leaderboard,
            classical_leaderboard=classical_leaderboard,
            tree_branch_metrics=tree_branch_metrics,
            classical_branch_metrics=classical_branch_metrics,
            comparison=comparison,
            feature_importance=feature_importance,
            output_dir=args.figures_dir,
        )

    outputs = {
        "tree_predictions": as_repo_relative(prediction_meta["path"], ROOT),
        "tree_leaderboard.csv": as_repo_relative(leaderboard_csv, ROOT),
        "tree_leaderboard.json": as_repo_relative(leaderboard_json, ROOT),
        "tree_branch_metrics.csv": as_repo_relative(branch_csv, ROOT),
        "tree_feature_importance.csv": as_repo_relative(importance_csv, ROOT),
        "tree_vs_classical_comparison.csv": as_repo_relative(comparison_csv, ROOT),
        "classical_test_leaderboard.csv": as_repo_relative(classical_test_csv, ROOT),
    }
    write_report(
        report_path=args.report_path,
        split_summaries=split_summaries,
        tree_leaderboard=tree_leaderboard,
        comparison=comparison,
        outputs=outputs,
        figures=figures,
        metadata=metadata,
        classical_full_leaderboard=classical_full_leaderboard,
    )

    summary = {
        "status": "ok",
        "prediction_output": outputs["tree_predictions"],
        "rows": int(len(tree_predictions)),
        "leaderboard": tree_leaderboard[
            [
                "dataset_id",
                "reconstruction_mode",
                "target_mode",
                "rank",
                "method",
                "model_family",
                "model_level",
                "mae",
                "dwell_mae",
                "weighted_energy_error",
            ]
        ].to_dict(orient="records"),
        "comparison": comparison.to_dict(orient="records"),
        "outputs": outputs,
        "figures": [as_repo_relative(path, ROOT) for path in figures],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
