from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ampere.data.canonical import write_dataframe_prefer_parquet
from ampere.features.splits import add_time_block_split
from ampere.neural.dwell_dataset import build_dwell_series, normalization_metadata
from ampere.neural.dwell_evaluate import (
    build_dwell_prediction_frame,
    compare_dwellnet_to_baselines,
    evaluate_dwellnet_predictions,
    load_matching_baselines,
    rank_dwellnet,
    summarize_ablation,
    validate_dwell_prediction_schema,
)
from ampere.neural.dwell_losses import DWELL_LOSS_VARIANTS
from ampere.neural.dwell_models import DWELL_MODEL_TYPES
from ampere.neural.dwell_train import DwellTrainConfig, predict_dwell_matrix, train_dwell_model
from ampere.neural.utils import TORCH_AVAILABLE, require_torch, resolve_device
from ampere.utils.paths import as_repo_relative


DATASET_LONG_INPUTS = {
    "rlc_sample": "rlc_sample_long.parquet",
    "appliance_8ch": "appliance_8ch_long.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scan-cycle-aligned DwellNet reconstruction experiments.")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_LONG_INPUTS), default=["appliance_8ch"])
    parser.add_argument("--target-modes", nargs="+", choices=["dwell_mean_power", "raw_signed_power"], default=["dwell_mean_power"])
    parser.add_argument("--models", nargs="+", choices=list(DWELL_MODEL_TYPES), default=list(DWELL_MODEL_TYPES))
    parser.add_argument("--window-cycles", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--output-modes", nargs="+", choices=["absolute", "residual_dwell"], default=["absolute", "residual_dwell"])
    parser.add_argument(
        "--loss-variants",
        "--losses",
        nargs="+",
        choices=list(DWELL_LOSS_VARIANTS),
        default=["dwell_supervised", "dwell_energy"],
        dest="loss_variants",
    )
    parser.add_argument("--normalization-modes", nargs="+", choices=["branchwise", "global"], default=["branchwise"])
    parser.add_argument("--no-time-features", action="store_true")
    parser.add_argument("--no-branch-embeddings", action="store_true")
    parser.add_argument("--no-branch-static", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--run-ablations", action="store_true")
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-val-windows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--mixed-precision", dest="mixed_precision", action="store_true", default=True)
    parser.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "outputs" / "data" / "processed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "reconstruction" / "dwellnet")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "outputs" / "figures" / "dwellnet")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "dwellnet_report.md")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def load_canonical_long(dataset_id: str, processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / DATASET_LONG_INPUTS[dataset_id]
    try:
        df = pd.read_parquet(path)
    except ImportError:
        csv_path = path.with_suffix(".csv")
        if not csv_path.exists():
            raise
        df = pd.read_csv(csv_path)
    return add_time_block_split(df)


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
    available = [column for column in columns if column in df.columns]
    table = df[available].copy()
    if max_rows is not None:
        table = table.head(max_rows)
    lines = [
        "| " + " | ".join(available) + " |",
        "| " + " | ".join(["---"] * len(available)) + " |",
    ]
    for row in table.itertuples(index=False):
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return lines


def default_epochs(args: argparse.Namespace) -> int:
    if args.epochs is not None:
        return int(args.epochs)
    if args.quick:
        return 5
    if args.full:
        return 120
    return 40


def default_patience(args: argparse.Namespace) -> int:
    if args.patience is not None:
        return int(args.patience)
    if args.quick:
        return 3
    if args.full:
        return 15
    return 8


def auto_batch_size(model_type: str, window_cycles: int, args: argparse.Namespace) -> int:
    if args.batch_size and args.batch_size > 0:
        return int(args.batch_size)
    device = resolve_device(args.device)
    if device.startswith("cuda"):
        if model_type == "dwell_transformer" or window_cycles >= 8:
            return 32
        return 64
    if model_type == "dwell_transformer" or window_cycles >= 8:
        return 16
    return 32


def config_for_run(
    args: argparse.Namespace,
    *,
    model_type: str,
    window_cycles: int,
    output_mode: str,
    normalization_mode: str,
    loss_variant: str,
    use_branch_embeddings: bool = True,
    include_time_features: bool = True,
    include_branch_static: bool = True,
    run_index: int = 0,
) -> DwellTrainConfig:
    if args.quick:
        cfg = DwellTrainConfig.quick(model_type=model_type, seed=args.seed + run_index)
    else:
        cfg = DwellTrainConfig(model_type=model_type, seed=args.seed + run_index)
    return replace(
        cfg,
        window_cycles=window_cycles,
        loss_variant=loss_variant,
        output_mode=output_mode,
        normalization_mode=normalization_mode,
        include_time_features=include_time_features,
        include_branch_static=include_branch_static,
        use_branch_embeddings=use_branch_embeddings,
        hidden_dim=32 if args.quick else args.hidden_dim,
        num_layers=1 if args.quick else args.num_layers,
        dropout=args.dropout,
        epochs=default_epochs(args),
        patience=default_patience(args),
        batch_size=auto_batch_size(model_type, window_cycles, args),
        max_train_windows=args.max_train_windows if args.max_train_windows is not None else (256 if args.quick else None),
        max_val_windows=args.max_val_windows if args.max_val_windows is not None else (128 if args.quick else None),
        grad_accum_steps=max(1, args.grad_accum_steps),
        device=args.device,
        mixed_precision=args.mixed_precision,
    )


def primary_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for model_type in args.models:
        for window_cycles in args.window_cycles:
            for output_mode in args.output_modes:
                for normalization_mode in args.normalization_modes:
                    for loss_variant in args.loss_variants:
                        rows.append(
                            {
                                "model_type": model_type,
                                "window_cycles": int(window_cycles),
                                "output_mode": output_mode,
                                "normalization_mode": normalization_mode,
                                "loss_variant": loss_variant,
                                "use_branch_embeddings": not args.no_branch_embeddings,
                                "include_time_features": not args.no_time_features,
                                "include_branch_static": not args.no_branch_static,
                                "grid_family": "primary",
                            }
                        )
    return rows


def ablation_grid(best_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    base = {
        "model_type": best_metadata["model_type"],
        "window_cycles": int(best_metadata["window_cycles"]),
        "output_mode": best_metadata["output_mode"],
        "normalization_mode": best_metadata["normalization_mode"],
        "loss_variant": best_metadata["loss_variant"],
        "use_branch_embeddings": bool(best_metadata["use_branch_embeddings"]),
        "include_time_features": bool(best_metadata["include_time_features"]),
        "include_branch_static": True,
    }
    return [
        {**base, "normalization_mode": "global", "grid_family": "ablation_global_norm"},
        {**base, "use_branch_embeddings": False, "grid_family": "ablation_no_branch_embeddings"},
        {**base, "include_time_features": False, "grid_family": "ablation_no_time_features"},
        {**base, "loss_variant": "dwell_full", "grid_family": "ablation_dwell_full"},
    ]


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def generate_figures(
    *,
    predictions: pd.DataFrame,
    leaderboard: pd.DataFrame,
    history: pd.DataFrame,
    comparison: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    plt = _load_matplotlib()
    if plt is None:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []
    if not leaderboard.empty:
        top = leaderboard.sort_values("mae").head(12)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(top)), top["mae"], color="#4c78a8")
        ax.set_xticks(range(len(top)), top["method"], rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("MAE")
        ax.set_title("DwellNet leaderboard")
        ax.grid(axis="y", alpha=0.25)
        path = output_dir / "dwellnet_leaderboard_mae.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(str(path))

    if not comparison.empty:
        best = comparison.sort_values("dwellnet_mae").head(1)
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = ["best DwellNet", "best tree"]
        values = [float(best["dwellnet_mae"].iloc[0]), float(best["tree_mae"].iloc[0])]
        ax.bar(labels, values, color=["#4c78a8", "#f58518"])
        ax.set_ylabel("MAE")
        ax.set_title("DwellNet vs matched tree baseline")
        ax.grid(axis="y", alpha=0.25)
        path = output_dir / "dwellnet_vs_tree_mae.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(str(path))

    if not history.empty:
        best_method = leaderboard.sort_values("mae").iloc[0]["method"] if not leaderboard.empty else history["method"].iloc[0]
        subset = history[history["method"].eq(best_method)]
        if not subset.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(subset["epoch"], subset["train_total"], label="train loss")
            ax.plot(subset["epoch"], subset["val_mae"], label="val MAE")
            ax.set_xlabel("epoch")
            ax.set_title(f"Training curve: {best_method}")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            path = output_dir / "best_dwellnet_training_curve.png"
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            figures.append(str(path))

    if not predictions.empty:
        best_method = leaderboard.sort_values("mae").iloc[0]["method"] if not leaderboard.empty else predictions["method"].iloc[0]
        subset = predictions[predictions["method"].eq(best_method)]
        branch_id = int(sorted(subset["branch_id"].unique())[0])
        branch = subset[subset["branch_id"].eq(branch_id)].sort_values("time_index").head(300)
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(branch["time_s"], branch["P_true"], color="black", linewidth=1.1, label="P_true")
        ax.plot(branch["time_s"], branch["P_hat"], linewidth=1.0, label="DwellNet")
        ax.set_title(f"Best DwellNet overlay, branch {branch_id}")
        ax.set_xlabel("time_s")
        ax.set_ylabel("power")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        path = output_dir / "best_dwellnet_overlay.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(str(path))
    return figures


def write_report(
    *,
    path: Path,
    args: argparse.Namespace,
    torch_version: str,
    device: str,
    cuda_available: bool,
    leaderboard: pd.DataFrame,
    branch_metrics: pd.DataFrame,
    ablation: pd.DataFrame,
    comparison: pd.DataFrame,
    baselines: pd.DataFrame,
    transition_metrics: pd.DataFrame,
    history: pd.DataFrame,
    output_paths: dict[str, str],
    figures: list[str],
    metadata: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best = leaderboard.sort_values("mae").head(1)
    best_comparison = comparison.sort_values("dwellnet_mae").head(1) if not comparison.empty else pd.DataFrame()
    beats_tree = bool(best_comparison["beats_tree_mae"].iloc[0]) if not best_comparison.empty else False
    lines = [
        "# DwellNet Reconstruction Report",
        "",
        "## Scope",
        "",
        "DwellNet is implemented as a post-dwell online reconstruction system. It uses completed current dwell observations and past observations only. It does not use future dwell measurements, `next_observed_value`, `time_to_next_seen`, or oracle `root_power`.",
        "",
        f"- PyTorch version: `{torch_version}`",
        f"- CUDA available: `{cuda_available}`",
        f"- Device used: `{device}`",
        f"- Full flag: `{args.full}`",
        f"- Quick flag: `{args.quick}`",
        "",
        "## Leaderboard",
        "",
    ]
    lines.extend(
        markdown_table(
            leaderboard.sort_values("mae"),
            [
                "dataset_id",
                "reconstruction_mode",
                "target_mode",
                "rank",
                "method",
                "neural_architecture",
                "window_cycles",
                "output_mode",
                "normalization_mode",
                "loss_variant",
                "mae",
                "rmse",
                "dwell_mae",
                "weighted_energy_error",
                "observed_point_mae",
                "unobserved_point_mae",
            ],
            max_rows=80,
        )
    )
    lines.extend(["", "## Matched Baselines", ""])
    lines.extend(markdown_table(baselines, ["baseline_family", "method", "mae", "dwell_mae", "weighted_energy_error"]))
    lines.extend(["", "## DwellNet vs Matched Tree", ""])
    lines.extend(markdown_table(comparison.sort_values("dwellnet_mae"), ["method", "dwellnet_mae", "tree_mae", "mae_delta_vs_tree", "beats_tree_mae"], max_rows=40))
    lines.extend(["", "## Ablation Summary", ""])
    lines.extend(markdown_table(ablation, ["ablation_axis", "value", "best_mae", "mean_mae", "runs", "delta_best_vs_overall_best"], max_rows=80))
    lines.extend(["", "## Transition And Stable Regions", ""])
    lines.extend(markdown_table(transition_metrics, ["method", "transition_mae", "stable_mae", "transition_rows", "stable_rows"], max_rows=40))

    lines.extend(["", "## Failure Analysis", ""])
    if beats_tree:
        lines.append("At least one fixed-grid DwellNet run, with its checkpoint selected by validation MAE, beats the matched tree MAE on the test split. This execution should still be treated as provisional unless it used the intended full GPU budget and is reviewed against dwell MAE, unobserved MAE, energy metrics, and validation/test gap.")
    else:
        lines.append("No DwellNet run in this execution beat the matched `window_dwell_extra_trees` baseline. This is a negative or inconclusive neural result for this run, not a final rejection of the DwellNet idea.")
        if not history.empty and not best.empty:
            best_method = best["method"].iloc[0]
            curve = history[history["method"].eq(best_method)]
            if not curve.empty:
                first_train = float(curve["train_total"].iloc[0])
                last_train = float(curve["train_total"].iloc[-1])
                first_val = float(curve["val_mae"].iloc[0])
                last_val = float(curve["val_mae"].iloc[-1])
                if last_train > 0.8 * first_train and last_val > 0.8 * first_val:
                    diagnosis = "underfit or insufficient optimization budget"
                elif last_train < 0.5 * first_train and last_val > first_val:
                    diagnosis = "overfit"
                else:
                    diagnosis = "architecture/feature/loss mismatch or limited run budget"
                lines.append(f"- Best-method train/val curve suggests: `{diagnosis}`.")
        if not branch_metrics.empty:
            worst = branch_metrics.sort_values("mae", ascending=False).head(3)
            labels = ", ".join(f"{row.branch_name} MAE {row.mae:.3f}" for row in worst.itertuples(index=False))
            lines.append(f"- Worst branch errors: {labels}.")
        lines.append("- Next fix: prioritize validation-selected DwellMLP/GRU families with branchwise normalization, residual output, time features, and longer GPU training before widening the architecture grid.")

    lines.extend(["", "## Outputs", ""])
    for label, output_path in output_paths.items():
        lines.append(f"- `{label}`: `{output_path}`")
    for figure in figures:
        lines.append(f"- figure: `{as_repo_relative(figure, ROOT)}`")
    lines.extend(["", "## Run Metadata", "", "```json", json.dumps(metadata, indent=2), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    if not TORCH_AVAILABLE:
        args.report_path.write_text("# DwellNet Reconstruction Report\n\nPyTorch is unavailable; DwellNet was skipped.\n", encoding="utf-8")
        print(json.dumps({"status": "skipped", "reason": "pytorch_unavailable"}, indent=2))
        return 0
    torch = require_torch()
    device = resolve_device(args.device)

    all_predictions: list[pd.DataFrame] = []
    all_history: list[pd.DataFrame] = []
    all_metadata: list[dict[str, Any]] = []
    normalization_rows: list[dict[str, Any]] = []
    series_for_metrics = None
    run_rows = primary_grid(args)

    run_index = 0
    for dataset_id in args.datasets:
        canonical = load_canonical_long(dataset_id, args.processed_dir)
        for target_mode in args.target_modes:
            primary_results: list[dict[str, Any]] = []
            for run in list(run_rows):
                cfg = config_for_run(args, run_index=run_index, **{key: run[key] for key in run if key != "grid_family"})
                series = build_dwell_series(
                    canonical,
                    target_mode=target_mode,
                    reconstruction_mode="online_safe",
                    output_mode=cfg.output_mode,
                    normalization_mode=cfg.normalization_mode,
                    include_time_features=cfg.include_time_features,
                    include_branch_static=cfg.include_branch_static,
                )
                if series_for_metrics is None:
                    series_for_metrics = series
                result = train_dwell_model(series, config=cfg, checkpoint_dir=checkpoint_dir)
                dwell_pred, _ = predict_dwell_matrix(result.model, series, config=cfg, split="test")
                predictions = build_dwell_prediction_frame(series, dwell_pred, metadata=result.metadata, split="test")
                validate_dwell_prediction_schema(predictions)
                all_predictions.append(predictions)
                history = pd.DataFrame(result.history)
                history["dataset_id"] = dataset_id
                history["target_mode"] = target_mode
                history["method"] = predictions["method"].iloc[0]
                history["model_type"] = cfg.model_type
                history["window_cycles"] = cfg.window_cycles
                history["output_mode"] = cfg.output_mode
                history["normalization_mode"] = cfg.normalization_mode
                history["loss_variant"] = cfg.loss_variant
                history["grid_family"] = run["grid_family"]
                all_history.append(history)
                meta = dict(result.metadata)
                meta["grid_family"] = run["grid_family"]
                all_metadata.append(meta)
                norm = normalization_metadata(series)
                norm["method"] = predictions["method"].iloc[0]
                normalization_rows.append(norm)
                primary_results.append(meta)
                run_index += 1

            if args.full or args.run_ablations:
                best_primary = sorted(primary_results, key=lambda row: row["best_val_mae"])[0]
                for run in ablation_grid(best_primary):
                    cfg = config_for_run(args, run_index=run_index, **{key: run[key] for key in run if key != "grid_family"})
                    series = build_dwell_series(
                        canonical,
                        target_mode=target_mode,
                        reconstruction_mode="online_safe",
                        output_mode=cfg.output_mode,
                        normalization_mode=cfg.normalization_mode,
                        include_time_features=cfg.include_time_features,
                        include_branch_static=cfg.include_branch_static,
                    )
                    result = train_dwell_model(series, config=cfg, checkpoint_dir=checkpoint_dir)
                    dwell_pred, _ = predict_dwell_matrix(result.model, series, config=cfg, split="test")
                    predictions = build_dwell_prediction_frame(series, dwell_pred, metadata=result.metadata, split="test")
                    validate_dwell_prediction_schema(predictions)
                    all_predictions.append(predictions)
                    history = pd.DataFrame(result.history)
                    history["dataset_id"] = dataset_id
                    history["target_mode"] = target_mode
                    history["method"] = predictions["method"].iloc[0]
                    history["model_type"] = cfg.model_type
                    history["window_cycles"] = cfg.window_cycles
                    history["output_mode"] = cfg.output_mode
                    history["normalization_mode"] = cfg.normalization_mode
                    history["loss_variant"] = cfg.loss_variant
                    history["grid_family"] = run["grid_family"]
                    all_history.append(history)
                    meta = dict(result.metadata)
                    meta["grid_family"] = run["grid_family"]
                    all_metadata.append(meta)
                    norm = normalization_metadata(series)
                    norm["method"] = predictions["method"].iloc[0]
                    normalization_rows.append(norm)
                    run_index += 1

    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    history = pd.concat(all_history, ignore_index=True) if all_history else pd.DataFrame()
    if series_for_metrics is None:
        raise RuntimeError("No DwellNet series was built")
    metrics, transition_metrics = evaluate_dwellnet_predictions(predictions, series_for_metrics)
    leaderboard = rank_dwellnet(metrics)
    branch_metrics = metrics.branch
    ablation = summarize_ablation(leaderboard)
    baseline_root = args.output_dir.parent
    tree_dir = baseline_root / "tree"
    constraint_dir = baseline_root / "constraints"
    classical_dir = baseline_root / "classical"
    if not tree_dir.exists():
        tree_dir = ROOT / "outputs" / "reconstruction" / "tree"
    if not constraint_dir.exists():
        constraint_dir = ROOT / "outputs" / "reconstruction" / "constraints"
    if not classical_dir.exists():
        classical_dir = ROOT / "outputs" / "reconstruction" / "classical"
    baselines = load_matching_baselines(
        dataset_id=args.datasets[0],
        target_mode=args.target_modes[0],
        reconstruction_mode="online_safe",
        tree_dir=tree_dir,
        constraint_dir=constraint_dir,
        classical_dir=classical_dir,
    )
    comparison = compare_dwellnet_to_baselines(leaderboard, baselines)

    prediction_meta = write_dataframe_prefer_parquet(predictions, args.output_dir / "dwellnet_predictions")
    leaderboard_csv = args.output_dir / "dwellnet_leaderboard.csv"
    leaderboard_json = args.output_dir / "dwellnet_leaderboard.json"
    branch_csv = args.output_dir / "dwellnet_branch_metrics.csv"
    ablation_csv = args.output_dir / "dwellnet_ablation_summary.csv"
    comparison_csv = args.output_dir / "dwellnet_vs_all_comparison.csv"
    history_csv = args.output_dir / "dwellnet_training_history.csv"
    transition_csv = args.output_dir / "dwellnet_transition_stable_metrics.csv"
    normalization_json = args.output_dir / "normalization_metadata.json"
    leaderboard.to_csv(leaderboard_csv, index=False)
    leaderboard.to_json(leaderboard_json, orient="records", indent=2)
    branch_metrics.to_csv(branch_csv, index=False)
    ablation.to_csv(ablation_csv, index=False)
    comparison.to_csv(comparison_csv, index=False)
    history.to_csv(history_csv, index=False)
    transition_metrics.to_csv(transition_csv, index=False)
    normalization_json.write_text(json.dumps(normalization_rows, indent=2), encoding="utf-8")

    figures: list[str] = []
    if not args.skip_plots:
        figures = generate_figures(
            predictions=predictions,
            leaderboard=leaderboard,
            history=history,
            comparison=comparison,
            output_dir=args.figures_dir,
        )

    output_paths = {
        "dwellnet_predictions": as_repo_relative(prediction_meta["path"], ROOT),
        "dwellnet_leaderboard.csv": as_repo_relative(leaderboard_csv, ROOT),
        "dwellnet_leaderboard.json": as_repo_relative(leaderboard_json, ROOT),
        "dwellnet_branch_metrics.csv": as_repo_relative(branch_csv, ROOT),
        "dwellnet_ablation_summary.csv": as_repo_relative(ablation_csv, ROOT),
        "dwellnet_vs_all_comparison.csv": as_repo_relative(comparison_csv, ROOT),
        "dwellnet_training_history.csv": as_repo_relative(history_csv, ROOT),
        "dwellnet_transition_stable_metrics.csv": as_repo_relative(transition_csv, ROOT),
        "normalization_metadata.json": as_repo_relative(normalization_json, ROOT),
        "checkpoints": as_repo_relative(checkpoint_dir, ROOT),
    }
    write_report(
        path=args.report_path,
        args=args,
        torch_version=torch.__version__,
        device=device,
        cuda_available=bool(torch.cuda.is_available()),
        leaderboard=leaderboard,
        branch_metrics=branch_metrics,
        ablation=ablation,
        comparison=comparison,
        baselines=baselines,
        transition_metrics=transition_metrics,
        history=history,
        output_paths=output_paths,
        figures=figures,
        metadata=all_metadata,
    )

    best = leaderboard.sort_values("mae").head(1)
    summary = {
        "status": "ok",
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": device,
        "runs": int(len(all_metadata)),
        "best": best[["dataset_id", "target_mode", "method", "mae", "dwell_mae", "weighted_energy_error"]].to_dict(
            orient="records"
        ),
        "comparison": comparison.sort_values("dwellnet_mae").head(3).to_dict(orient="records"),
        "outputs": output_paths,
        "figures": [as_repo_relative(path, ROOT) for path in figures],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
