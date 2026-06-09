from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ampere.data.canonical import write_dataframe_prefer_parquet
from ampere.features.splits import add_time_block_split
from ampere.neural.observer_dataset import (
    OBSERVER_LOSS_VARIANTS,
    OBSERVER_MODEL_TYPES,
    build_observer_series,
    observer_normalization_metadata,
)
from ampere.neural.observer_evaluate import (
    OBSERVER_EVAL_MODES,
    build_gain_frame,
    build_observer_prediction_frame,
    compare_observer_to_references,
    evaluate_observer_predictions,
    gain_diagnostics,
    observer_method_name,
    rank_observer,
    reference_baselines,
    validate_observer_prediction_schema,
)
from ampere.neural.observer_losses import OBSERVER_TEACHER_MASKS, OBSERVER_TEACHER_SPACES
from ampere.neural.observer_train import (
    ObserverTrainConfig,
    load_observer_model_checkpoint,
    observer_checkpoint_path,
    predict_observer_matrix,
    train_observer_model,
)
from ampere.neural.utils import TORCH_AVAILABLE, require_torch, resolve_device
from ampere.utils.paths import as_repo_relative


DATASET_LONG_INPUTS = {
    "rlc_sample": "rlc_sample_long.parquet",
    "appliance_8ch": "appliance_8ch_long.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DwellObserver / DwellObserver-T experiments.")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_LONG_INPUTS), default=["appliance_8ch"])
    parser.add_argument("--target-modes", nargs="+", choices=["dwell_mean_power", "raw_signed_power"], default=["dwell_mean_power"])
    parser.add_argument("--models", nargs="+", choices=list(OBSERVER_MODEL_TYPES), default=["observer_gru"])
    parser.add_argument("--window-cycles", nargs="+", type=int, default=[8])
    parser.add_argument("--loss-variants", "--losses", nargs="+", choices=list(OBSERVER_LOSS_VARIANTS), default=["observer_supervised"], dest="loss_variants")
    parser.add_argument("--transition-weights", nargs="+", type=float, default=[0.5])
    parser.add_argument("--gain-lambdas", nargs="+", type=float, default=[0.0])
    parser.add_argument("--use-ema-target", action="store_true")
    parser.add_argument("--ema-taus", nargs="+", type=float, default=[0.005])
    parser.add_argument("--teacher-lambdas", nargs="+", type=float, default=[0.01])
    parser.add_argument("--teacher-mask", nargs="+", choices=list(OBSERVER_TEACHER_MASKS), default=["unobserved_only"])
    parser.add_argument("--teacher-space", nargs="+", choices=list(OBSERVER_TEACHER_SPACES), default=["residual_delta"])
    parser.add_argument("--teacher-warmup-epochs", type=int, default=0)
    parser.add_argument("--teacher-ramp-epochs", type=int, default=0)
    parser.add_argument("--teacher-perturbation", nargs="+", choices=["none", "light_mask_noise"], default=["none"])
    parser.add_argument("--eval-ema-teacher", action="store_true")
    parser.add_argument("--eval-student-teacher-average", action="store_true")
    parser.add_argument("--normalization-modes", nargs="+", choices=["branchwise", "global"], default=["branchwise"])
    parser.add_argument("--output-modes", nargs="+", choices=["residual_dwell"], default=["residual_dwell"])
    parser.add_argument("--no-time-features", action="store_true")
    parser.add_argument("--no-branch-embeddings", action="store_true")
    parser.add_argument("--no-branch-static", action="store_true")
    parser.add_argument("--no-hard-observed-projection", dest="hard_observed_projection", action="store_false", default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-val-windows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--gain-bound", type=float, default=2.0)
    parser.add_argument("--mixed-precision", dest="mixed_precision", action="store_true", default=True)
    parser.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "outputs" / "data" / "processed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "reconstruction" / "dwellobserver_t")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "outputs" / "figures" / "dwellobserver_t")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "dwellobserver_t_report.md")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--reuse-checkpoints", action="store_true")
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
    lines = ["| " + " | ".join(available) + " |", "| " + " | ".join(["---"] * len(available)) + " |"]
    for row in table.itertuples(index=False):
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return lines


def seed_repeatability_summary(leaderboard: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    required = {"observer_config_method", "observer_eval_mode", "seed", "dwell_mae", "unobserved_point_mae", "weighted_energy_error"}
    if leaderboard.empty or not required.issubset(leaderboard.columns) or leaderboard["seed"].nunique() < 2:
        return pd.DataFrame()
    transition_subset = (
        transitions[["method", "transition_mae", "stable_mae"]]
        if not transitions.empty
        else pd.DataFrame(columns=["method", "transition_mae", "stable_mae"])
    )
    merged = leaderboard.merge(transition_subset, on="method", how="left")
    rows = []
    for keys, group in merged.groupby(["observer_config_method", "observer_eval_mode"], sort=False):
        rows.append(
            {
                "observer_config_method": keys[0],
                "observer_eval_mode": keys[1],
                "seeds": ",".join(str(int(seed)) for seed in sorted(group["seed"].dropna().unique())),
                "dwell_mae_mean": float(group["dwell_mae"].mean()),
                "dwell_mae_std": float(group["dwell_mae"].std(ddof=0)),
                "unobserved_mae_mean": float(group["unobserved_point_mae"].mean()),
                "unobserved_mae_std": float(group["unobserved_point_mae"].std(ddof=0)),
                "weighted_energy_error_mean": float(group["weighted_energy_error"].mean()),
                "weighted_energy_error_std": float(group["weighted_energy_error"].std(ddof=0)),
                "transition_mae_mean": float(group["transition_mae"].mean()),
                "transition_mae_std": float(group["transition_mae"].std(ddof=0)),
                "stable_mae_mean": float(group["stable_mae"].mean()),
                "stable_mae_std": float(group["stable_mae"].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def auto_batch_size(model_type: str, window_cycles: int, args: argparse.Namespace) -> int:
    if args.batch_size and args.batch_size > 0:
        return int(args.batch_size)
    device = resolve_device(args.device)
    if device.startswith("cuda"):
        return 32 if window_cycles >= 8 or model_type == "observer_gru" else 64
    return 16 if window_cycles >= 8 or model_type == "observer_gru" else 32


def config_for_run(
    args: argparse.Namespace,
    *,
    seed: int,
    model_type: str,
    window_cycles: int,
    output_mode: str,
    normalization_mode: str,
    loss_variant: str,
    transition_weight: float,
    gain_lambda: float,
    use_ema_target: bool,
    ema_tau: float,
    teacher_lambda: float,
    teacher_mask: str,
    teacher_space: str,
    teacher_perturbation: str,
    run_index: int,
) -> ObserverTrainConfig:
    if args.quick:
        cfg = ObserverTrainConfig.quick(model_type=model_type, seed=seed)
    else:
        cfg = ObserverTrainConfig(model_type=model_type, seed=seed)
    return replace(
        cfg,
        window_cycles=window_cycles,
        output_mode=output_mode,
        normalization_mode=normalization_mode,
        loss_variant=loss_variant,
        include_time_features=not args.no_time_features,
        include_branch_static=not args.no_branch_static,
        use_branch_embeddings=not args.no_branch_embeddings,
        hidden_dim=32 if args.quick else args.hidden_dim,
        num_layers=1 if args.quick else args.num_layers,
        dropout=args.dropout,
        gain_bound=args.gain_bound,
        transition_weight=float(transition_weight),
        gain_lambda=float(gain_lambda),
        use_ema_target=bool(use_ema_target),
        ema_tau=float(ema_tau),
        teacher_lambda=float(teacher_lambda),
        teacher_mask=teacher_mask,
        teacher_space=teacher_space,
        teacher_warmup_epochs=max(0, int(args.teacher_warmup_epochs)),
        teacher_ramp_epochs=max(0, int(args.teacher_ramp_epochs)),
        teacher_perturbation=teacher_perturbation,
        epochs=5 if args.quick else args.epochs,
        patience=3 if args.quick else args.patience,
        batch_size=auto_batch_size(model_type, window_cycles, args),
        max_train_windows=args.max_train_windows if args.max_train_windows is not None else (256 if args.quick else None),
        max_val_windows=args.max_val_windows if args.max_val_windows is not None else (128 if args.quick else None),
        grad_accum_steps=max(1, args.grad_accum_steps),
        device=args.device,
        mixed_precision=args.mixed_precision,
        hard_observed_projection=args.hard_observed_projection,
    )


def model_uses_learned_gain(model_type: str) -> bool:
    return model_type in {
        "observer_gru",
        "observer_mlp",
        "observer_mlp_learned_selected_gain",
        "observer_mlp_sparse_offbranch_gain",
    }


def primary_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for model_type in args.models:
        for window_cycles in args.window_cycles:
            for output_mode in args.output_modes:
                for normalization_mode in args.normalization_modes:
                    for loss_variant in args.loss_variants:
                        transition_weights = args.transition_weights if loss_variant == "observer_transition" else [0.0]
                        gain_lambdas = args.gain_lambdas if model_uses_learned_gain(model_type) else [0.0]
                        for transition_weight in transition_weights:
                            for gain_lambda in gain_lambdas:
                                ema_taus = args.ema_taus if args.use_ema_target else [0.0]
                                teacher_lambdas = args.teacher_lambdas if args.use_ema_target else [0.0]
                                teacher_masks = args.teacher_mask if args.use_ema_target else ["none"]
                                teacher_spaces = args.teacher_space if args.use_ema_target else ["none"]
                                teacher_perturbations = args.teacher_perturbation if args.use_ema_target else ["none"]
                                for ema_tau in ema_taus:
                                    for teacher_lambda in teacher_lambdas:
                                        for teacher_mask in teacher_masks:
                                            for teacher_space in teacher_spaces:
                                                for teacher_perturbation in teacher_perturbations:
                                                    rows.append(
                                                        {
                                                            "model_type": model_type,
                                                            "window_cycles": int(window_cycles),
                                                            "output_mode": output_mode,
                                                            "normalization_mode": normalization_mode,
                                                            "loss_variant": loss_variant,
                                                            "transition_weight": float(transition_weight),
                                                            "gain_lambda": float(gain_lambda),
                                                            "use_ema_target": bool(args.use_ema_target),
                                                            "ema_tau": float(ema_tau),
                                                            "teacher_lambda": float(teacher_lambda),
                                                            "teacher_mask": teacher_mask,
                                                            "teacher_space": teacher_space,
                                                            "teacher_perturbation": teacher_perturbation,
                                                        }
                                                    )
    return rows


def evaluation_modes_for_args(args: argparse.Namespace) -> list[str]:
    modes = ["student"]
    if args.use_ema_target and args.eval_ema_teacher:
        modes.append("ema_teacher")
    if args.use_ema_target and args.eval_student_teacher_average:
        modes.append("student_teacher_average")
    return [mode for mode in modes if mode in OBSERVER_EVAL_MODES]


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
    gain_by_selected: pd.DataFrame,
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
        ax.bar(range(len(top)), top["mae"], color="#2563eb")
        ax.set_xticks(range(len(top)), top["method"], rotation=55, ha="right", fontsize=7)
        ax.set_ylabel("Test dwell MAE")
        ax.set_title("DwellObserver leaderboard")
        ax.grid(axis="y", alpha=0.25)
        path = output_dir / "observer_leaderboard_mae.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(str(path))

    if not comparison.empty:
        best_method = leaderboard.sort_values("mae").iloc[0]["method"]
        subset = comparison[comparison["observer_method"].eq(best_method)]
        subset = subset[subset["baseline_family"].isin(["dwellnet_validated", "tree", "residual_prior_zoh"])]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        labels = ["observer"] + subset["baseline_family"].tolist()
        values = [float(leaderboard.sort_values("mae").iloc[0]["dwell_mae"])] + subset["baseline_dwell_mae"].astype(float).tolist()
        ax.bar(labels, values, color=["#2563eb", "#16a34a", "#f59e0b", "#6b7280"][: len(labels)])
        ax.set_ylabel("Dwell MAE")
        ax.set_title("Best DwellObserver vs references")
        ax.grid(axis="y", alpha=0.25)
        path = output_dir / "observer_vs_references.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(str(path))

    if not history.empty and not leaderboard.empty:
        best_method = leaderboard.sort_values("mae").iloc[0]["method"]
        subset = history[history["method"].eq(best_method)]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(subset["epoch"], subset["train_total"], label="train loss")
        ax.plot(subset["epoch"], subset["val_mae"], label="val MAE")
        ax.set_xlabel("epoch")
        ax.set_title(f"Training curve: {best_method}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        path = output_dir / "best_observer_training_curve.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(str(path))

    if not gain_by_selected.empty:
        best_method = leaderboard.sort_values("mae").iloc[0]["method"] if not leaderboard.empty else gain_by_selected["method"].iloc[0]
        heat = gain_by_selected[gain_by_selected["method"].eq(best_method)]
        pivot = heat.pivot_table(index="selected_branch", columns="branch_id", values="mean_gain", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="coolwarm")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_xlabel("Updated branch")
        ax.set_ylabel("Selected branch")
        ax.set_title("Mean observer gain")
        fig.colorbar(im, ax=ax, fraction=0.046)
        path = output_dir / "observer_gain_heatmap.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(str(path))

    if not predictions.empty and not leaderboard.empty:
        best_method = leaderboard.sort_values("mae").iloc[0]["method"]
        subset = predictions[predictions["method"].eq(best_method)]
        branch_id = int(subset.groupby("branch_id")["P_true"].std().sort_values(ascending=False).index[0])
        branch = subset[subset["branch_id"].eq(branch_id)].sort_values("time_index").head(300)
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(branch["time_s"], branch["P_true"], color="black", linewidth=1.1, label="P_true")
        ax.plot(branch["time_s"], branch["P_hat"], linewidth=1.0, label="DwellObserver")
        ax.set_title(f"Best DwellObserver overlay, branch {branch_id}")
        ax.set_xlabel("time_s")
        ax.set_ylabel("power")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        path = output_dir / "best_observer_overlay.png"
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
    validation_leaderboard: pd.DataFrame,
    leaderboard: pd.DataFrame,
    branch_metrics: pd.DataFrame,
    transition_metrics: pd.DataFrame,
    val_test_gap: pd.DataFrame,
    history: pd.DataFrame,
    baselines: pd.DataFrame,
    comparison: pd.DataFrame,
    gain_summary: pd.DataFrame,
    gain_by_selected: pd.DataFrame,
    repeatability: pd.DataFrame,
    output_paths: dict[str, str],
    figures: list[str],
    metadata: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_val = validation_leaderboard.sort_values("mae").head(1)
    selected_method = str(selected_val["method"].iloc[0]) if not selected_val.empty else ""
    selected_test = leaderboard[leaderboard["method"].eq(selected_method)].head(1)
    noema = baselines[baselines["baseline_family"].eq("stage3c_r1_no_ema")].head(1)
    selected_transition = (
        transition_metrics[transition_metrics["method"].eq(selected_method)].head(1)
        if not transition_metrics.empty
        else pd.DataFrame()
    )
    selected_dwell = float(selected_test["dwell_mae"].iloc[0]) if not selected_test.empty else float("nan")
    selected_energy = float(selected_test["weighted_energy_error"].iloc[0]) if not selected_test.empty else float("nan")
    selected_transition_mae = (
        float(selected_transition["transition_mae"].iloc[0]) if not selected_transition.empty else float("nan")
    )
    noema_dwell = float(noema["dwell_mae"].iloc[0]) if not noema.empty else float("nan")
    noema_energy = float(noema["weighted_energy_error"].iloc[0]) if not noema.empty else float("nan")
    noema_transition = float(noema["transition_mae"].iloc[0]) if not noema.empty else float("nan")
    dwell_delta = selected_dwell - noema_dwell
    transition_delta = selected_transition_mae - noema_transition
    energy_ratio = selected_energy / noema_energy if noema_energy > 0.0 else float("nan")
    promising = bool(
        (pd.notna(dwell_delta) and dwell_delta < 0.0 and pd.notna(transition_delta) and transition_delta <= 0.0)
        or (pd.notna(transition_delta) and transition_delta <= -3.0 and pd.notna(dwell_delta) and dwell_delta <= 0.75)
        or (pd.notna(energy_ratio) and energy_ratio <= 0.80 and pd.notna(dwell_delta) and dwell_delta <= 0.75)
    )
    lines = [
        "# DwellObserver-T Stage 3D-R1 Refined Report",
        "",
        "## Scope",
        "",
        "This run tests refined DwellObserver-T EMA target stabilization as a supervised consistency regularizer. It does not include Stage 5 scaling, rollout, uncertainty heads, adaptive sampling, RL, oracle root_power, future features, or raw Data edits.",
        "",
        f"- PyTorch version: `{torch_version}`",
        f"- CUDA available: `{cuda_available}`",
        f"- Device used: `{device}`",
        f"- Hard observed projection: `{args.hard_observed_projection}`",
        f"- EMA target enabled: `{args.use_ema_target}`",
        f"- EMA taus requested: `{args.ema_taus}`",
        f"- Teacher lambdas requested: `{args.teacher_lambdas}`",
        f"- Teacher masks requested: `{args.teacher_mask}`",
        f"- Teacher spaces requested: `{args.teacher_space}`",
        f"- Teacher warmup/ramp epochs: `{args.teacher_warmup_epochs}` / `{args.teacher_ramp_epochs}`",
        f"- Teacher perturbations requested: `{args.teacher_perturbation}`",
        f"- Evaluation modes: `{evaluation_modes_for_args(args)}`",
        f"- Seeds: `{args.seeds if args.seeds is not None else [args.seed]}`",
        f"- Command: `{' '.join(sys.argv)}`",
        "",
        "## Teacher Design",
        "",
        "- `unobserved_only` masks teacher consistency to branches without the post-dwell observation.",
        "- `stale_unobserved_weighted` additionally weights unobserved branches by causal pre-dwell staleness and normalizes active weights near mean 1.",
        "- `residual_delta` matches `posterior_scaled - pre_base_scaled`, so the teacher stabilizes residual correction rather than the known selected-branch projection.",
        "- Warmup keeps teacher loss at zero; ramp linearly increases the effective lambda.",
        "- The EMA teacher is frozen and updated only with `theta_minus <- tau * theta + (1 - tau) * theta_minus` after optimizer steps.",
        "",
        "## Validation Leaderboard",
        "",
    ]
    lines.extend(
        markdown_table(
            validation_leaderboard.sort_values("mae"),
            [
                "dataset_id",
                "reconstruction_mode",
                "target_mode",
                "rank",
                "method",
                "neural_architecture",
                "mae",
                "dwell_mae",
                "weighted_energy_error",
                "observed_point_mae",
                "unobserved_point_mae",
            ],
            max_rows=80,
        )
    )
    lines.extend(["", "## Training Best-Val Checkpoints", ""])
    train_rows = []
    for meta in metadata:
        train_rows.append(
            {
                "method": meta["method"],
                "best_val_mae": meta["best_val_mae"],
                "seed": meta.get("seed", 0),
                "epochs_completed": meta["epochs_completed"],
                "train_windows": meta["train_windows"],
                "val_windows": meta["val_windows"],
                "transition_weight": meta.get("transition_weight", 0.0),
                "gain_lambda": meta.get("gain_lambda", 0.0),
                "use_ema_target": meta.get("use_ema_target", False),
                "ema_tau": meta.get("ema_tau", 0.0),
                "teacher_lambda": meta.get("teacher_lambda", 0.0),
                "teacher_mask": meta.get("teacher_mask", "none"),
                "teacher_space": meta.get("teacher_space", "none"),
                "teacher_perturbation": meta.get("teacher_perturbation", "none"),
            }
        )
    lines.extend(
        markdown_table(
            pd.DataFrame(train_rows).sort_values("best_val_mae"),
            [
                "method",
                "best_val_mae",
                "seed",
                "epochs_completed",
                "train_windows",
                "val_windows",
                "transition_weight",
                "gain_lambda",
                "use_ema_target",
                "ema_tau",
                "teacher_lambda",
                "teacher_mask",
                "teacher_space",
                "teacher_perturbation",
            ],
        )
    )
    lines.extend(["", "## Test Leaderboard", ""])
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
                "mae",
                "dwell_mae",
                "weighted_energy_error",
                "observed_point_mae",
                "unobserved_point_mae",
            ],
            max_rows=60,
        )
    )
    lines.extend(["", "## Validation/Test Gap", ""])
    lines.extend(
        markdown_table(
            val_test_gap.sort_values("validation_mae"),
            [
                "method",
                "validation_dwell_mae",
                "dwell_mae",
                "validation_test_dwell_gap",
                "weighted_energy_error",
                "unobserved_point_mae",
            ],
            max_rows=80,
        )
    )
    lines.extend(["", "## Reference Baselines", ""])
    lines.extend(markdown_table(baselines, ["baseline_family", "method", "dwell_mae", "unobserved_mae", "weighted_energy_error", "transition_mae", "stable_mae"]))
    lines.extend(["", "## DwellObserver Vs References", ""])
    lines.extend(
        markdown_table(
            comparison[
                comparison["baseline_family"].isin(
                    ["stage3c_r1_no_ema", "dwellnet_validated", "dwellnet_secondary", "tree", "residual_prior_zoh"]
                )
            ].sort_values(
                ["observer_method", "baseline_family"]
            ),
            [
                "observer_method",
                "baseline_family",
                "dwell_mae_delta",
                "unobserved_mae_delta",
                "energy_error_delta",
                "transition_mae_delta",
            ],
            max_rows=80,
        )
    )
    lines.extend(["", "## Transition And Stable Regions", ""])
    lines.extend(markdown_table(transition_metrics, ["method", "transition_mae", "stable_mae", "transition_rows", "stable_rows"], max_rows=60))
    lines.extend(["", "## Seed Repeatability", ""])
    lines.extend(
        markdown_table(
            repeatability.sort_values("dwell_mae_mean") if not repeatability.empty else repeatability,
            [
                "observer_config_method",
                "observer_eval_mode",
                "seeds",
                "dwell_mae_mean",
                "dwell_mae_std",
                "unobserved_mae_mean",
                "unobserved_mae_std",
                "weighted_energy_error_mean",
                "weighted_energy_error_std",
                "transition_mae_mean",
                "transition_mae_std",
                "stable_mae_mean",
                "stable_mae_std",
            ],
            max_rows=80,
        )
    )
    lines.extend(["", "## Per-Branch Errors", ""])
    lines.extend(markdown_table(branch_metrics.sort_values("mae", ascending=False), ["method", "branch_id", "branch_name", "mae", "dwell_mae", "abs_energy_error_Wh"], max_rows=40))
    lines.extend(["", "## Gain Diagnostics", ""])
    lines.extend(markdown_table(gain_summary, ["method", "mean_selected_branch_gain", "mean_off_branch_abs_gain", "max_abs_gain", "mean_abs_innovation_scaled"], max_rows=40))
    lines.extend(["", "## Gain By Selected Branch", ""])
    lines.extend(markdown_table(gain_by_selected, ["method", "selected_branch", "branch_id", "is_selected_branch", "mean_gain", "mean_abs_gain"], max_rows=80))
    lines.extend(["", "## Interpretation", ""])
    if promising:
        lines.append(
            "The validation-selected refined DwellObserver-T row beats or complements the no-EMA Stage 3C-R1 observer by the predeclared success criteria. Treat it as promising, then confirm with seed repeatability before promotion."
        )
    else:
        lines.append(
            "The validation-selected refined DwellObserver-T row does not clearly beat or complement the no-EMA Stage 3C-R1 observer. Keep DwellObserver-T as an ablation/future-work idea and move next to Stage 5 scaling."
        )
    lines.append(f"- Validation-selected method: `{selected_method}`.")
    lines.append(f"- Delta vs no-EMA Stage 3C-R1 dwell MAE: `{dwell_delta:.4f}`.")
    lines.append(f"- Delta vs no-EMA Stage 3C-R1 transition MAE: `{transition_delta:.4f}`.")
    lines.append(f"- Weighted energy ratio vs no-EMA Stage 3C-R1: `{energy_ratio:.4f}`.")
    lines.append("- The no-EMA Stage 3C-R1 `observer_mlp_prior_only` remains the main observer unless the repeated R1 result satisfies the success criteria.")
    lines.append("- EMA target stabilization is supervised teacher consistency only, not DQN, Q-learning, RL scheduling, or adaptive sampling.")
    lines.append("- DwellMLP, DwellFormer no-time, the tree baseline, and the no-EMA observer are included as comparison anchors.")
    lines.append("- Observed-point MAE is secondary because selected branches are known post-dwell and hard-projected in the default evaluation.")
    lines.append("- Primary interpretation should use dwell MAE, unobserved MAE, transition/stable MAE, weighted energy error, and gain diagnostics together.")
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
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir is not None else args.output_dir / "checkpoints"
    if not TORCH_AVAILABLE:
        args.report_path.write_text("# DwellObserver-T Stage 3D Report\n\nPyTorch unavailable; skipped.\n", encoding="utf-8")
        print(json.dumps({"status": "skipped", "reason": "pytorch_unavailable"}, indent=2))
        return 0
    torch = require_torch()
    device = resolve_device(args.device)

    all_predictions: list[pd.DataFrame] = []
    all_val_predictions: list[pd.DataFrame] = []
    all_overall_metrics: list[pd.DataFrame] = []
    all_branch_metrics: list[pd.DataFrame] = []
    all_transition_metrics: list[pd.DataFrame] = []
    all_val_overall_metrics: list[pd.DataFrame] = []
    all_val_transition_metrics: list[pd.DataFrame] = []
    all_history: list[pd.DataFrame] = []
    all_gain_frames: list[pd.DataFrame] = []
    all_metadata: list[dict[str, Any]] = []
    normalization_rows: list[dict[str, Any]] = []
    series_for_metrics = None
    run_index = 0
    seeds = [int(value) for value in (args.seeds if args.seeds is not None else [args.seed])]
    eval_modes = evaluation_modes_for_args(args)
    use_seed_suffix = len(seeds) > 1

    for dataset_id in args.datasets:
        canonical = load_canonical_long(dataset_id, args.processed_dir)
        for target_mode in args.target_modes:
            for seed in seeds:
                for run in primary_grid(args):
                    cfg = config_for_run(args, seed=seed, run_index=run_index, **run)
                    series = build_observer_series(
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
                    checkpoint_path = observer_checkpoint_path(series, cfg, checkpoint_dir)
                    if args.reuse_checkpoints and checkpoint_path.exists():
                        result = load_observer_model_checkpoint(series, config=cfg, checkpoint_path=checkpoint_path)
                    else:
                        result = train_observer_model(series, config=cfg, checkpoint_dir=checkpoint_dir)
                    train_meta = dict(result.metadata)
                    train_meta["method_seed_suffix"] = use_seed_suffix
                    student_meta = dict(train_meta)
                    student_meta["observer_eval_mode"] = "student"
                    student_method = observer_method_name(student_meta)

                    for eval_mode in eval_modes:
                        eval_meta = dict(train_meta)
                        eval_meta["observer_eval_mode"] = eval_mode
                        dwell_pred_val, _prior_val, _gains_val, _innovations_val, _ends_val = predict_observer_matrix(
                            result.model,
                            series,
                            config=cfg,
                            split="val",
                            target_model=result.ema_target_model,
                            eval_mode=eval_mode,
                        )
                        val_predictions = build_observer_prediction_frame(series, dwell_pred_val, metadata=eval_meta, split="val")
                        validate_observer_prediction_schema(val_predictions)
                        val_metric_tables, val_transition = evaluate_observer_predictions(val_predictions, series)
                        all_val_overall_metrics.append(val_metric_tables.overall)
                        all_val_transition_metrics.append(val_transition)
                        all_val_predictions.append(val_predictions)

                        dwell_pred, _prior, gains, innovations, _ends = predict_observer_matrix(
                            result.model,
                            series,
                            config=cfg,
                            split="test",
                            target_model=result.ema_target_model,
                            eval_mode=eval_mode,
                        )
                        predictions = build_observer_prediction_frame(series, dwell_pred, metadata=eval_meta, split="test")
                        validate_observer_prediction_schema(predictions)
                        metric_tables, transition_table = evaluate_observer_predictions(predictions, series)
                        all_overall_metrics.append(metric_tables.overall)
                        all_branch_metrics.append(metric_tables.branch)
                        all_transition_metrics.append(transition_table)
                        gain_frame = build_gain_frame(series, gains, innovations, metadata=eval_meta, split="test")
                        all_predictions.append(predictions)
                        all_gain_frames.append(gain_frame)

                    history = pd.DataFrame(result.history)
                    history["dataset_id"] = dataset_id
                    history["target_mode"] = target_mode
                    history["method"] = student_method
                    history["model_type"] = cfg.model_type
                    history["window_cycles"] = cfg.window_cycles
                    history["output_mode"] = cfg.output_mode
                    history["normalization_mode"] = cfg.normalization_mode
                    history["loss_variant"] = cfg.loss_variant
                    history["use_ema_target"] = cfg.use_ema_target
                    history["ema_tau"] = cfg.ema_tau if cfg.use_ema_target else 0.0
                    history["teacher_lambda"] = cfg.teacher_lambda if cfg.use_ema_target else 0.0
                    history["teacher_mask"] = cfg.teacher_mask if cfg.use_ema_target else "none"
                    history["teacher_space"] = cfg.teacher_space if cfg.use_ema_target else "none"
                    history["teacher_perturbation"] = cfg.teacher_perturbation if cfg.use_ema_target else "none"
                    history["seed"] = cfg.seed
                    all_history.append(history)
                    meta = dict(train_meta)
                    meta["method"] = student_method
                    all_metadata.append(meta)
                    norm = observer_normalization_metadata(series)
                    norm["method"] = student_method
                    norm["seed"] = cfg.seed
                    normalization_rows.append(norm)
                    run_index += 1

    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    val_predictions = pd.concat(all_val_predictions, ignore_index=True) if all_val_predictions else pd.DataFrame()
    history = pd.concat(all_history, ignore_index=True) if all_history else pd.DataFrame()
    gain_frame = pd.concat(all_gain_frames, ignore_index=True) if all_gain_frames else pd.DataFrame()
    if series_for_metrics is None:
        raise RuntimeError("No DwellObserver series was built")
    overall_metrics = pd.concat(all_overall_metrics, ignore_index=True) if all_overall_metrics else pd.DataFrame()
    branch_metrics = pd.concat(all_branch_metrics, ignore_index=True) if all_branch_metrics else pd.DataFrame()
    transition_metrics = pd.concat(all_transition_metrics, ignore_index=True) if all_transition_metrics else pd.DataFrame()
    val_overall_metrics = pd.concat(all_val_overall_metrics, ignore_index=True) if all_val_overall_metrics else pd.DataFrame()
    val_transition_metrics = (
        pd.concat(all_val_transition_metrics, ignore_index=True) if all_val_transition_metrics else pd.DataFrame()
    )
    leaderboard = rank_observer(overall_metrics)
    validation_leaderboard = rank_observer(val_overall_metrics)
    gain_summary, gain_by_selected = gain_diagnostics(gain_frame)
    baselines = reference_baselines()
    comparison = compare_observer_to_references(leaderboard, transition_metrics)
    repeatability = seed_repeatability_summary(leaderboard, transition_metrics)
    validation_by_method = validation_leaderboard[["method", "mae", "dwell_mae"]].rename(
        columns={"mae": "validation_mae", "dwell_mae": "validation_dwell_mae"}
    )
    val_test_gap = leaderboard.merge(validation_by_method, on="method", how="left")
    val_test_gap["validation_test_dwell_gap"] = val_test_gap["dwell_mae"] - val_test_gap["validation_dwell_mae"]

    prediction_meta = write_dataframe_prefer_parquet(predictions, args.output_dir / "observer_predictions")
    validation_prediction_meta = write_dataframe_prefer_parquet(val_predictions, args.output_dir / "observer_validation_predictions")
    leaderboard_csv = args.output_dir / "observer_leaderboard.csv"
    validation_leaderboard_csv = args.output_dir / "observer_validation_leaderboard.csv"
    leaderboard_json = args.output_dir / "observer_leaderboard.json"
    branch_csv = args.output_dir / "observer_branch_metrics.csv"
    transition_csv = args.output_dir / "observer_transition_stable_metrics.csv"
    validation_transition_csv = args.output_dir / "observer_validation_transition_stable_metrics.csv"
    val_test_gap_csv = args.output_dir / "observer_validation_test_gap.csv"
    history_csv = args.output_dir / "observer_training_history.csv"
    gain_csv = args.output_dir / "observer_gain_frame.csv"
    gain_summary_csv = args.output_dir / "observer_gain_diagnostics.csv"
    gain_by_selected_csv = args.output_dir / "observer_gain_by_selected_branch.csv"
    comparison_csv = args.output_dir / "observer_vs_baselines.csv"
    baselines_csv = args.output_dir / "observer_reference_baselines.csv"
    repeatability_csv = args.output_dir / "observer_seed_repeatability.csv"
    normalization_json = args.output_dir / "normalization_metadata.json"

    leaderboard.to_csv(leaderboard_csv, index=False)
    validation_leaderboard.to_csv(validation_leaderboard_csv, index=False)
    leaderboard.to_json(leaderboard_json, orient="records", indent=2)
    branch_metrics.to_csv(branch_csv, index=False)
    transition_metrics.to_csv(transition_csv, index=False)
    val_transition_metrics.to_csv(validation_transition_csv, index=False)
    val_test_gap.to_csv(val_test_gap_csv, index=False)
    history.to_csv(history_csv, index=False)
    gain_frame.to_csv(gain_csv, index=False)
    gain_summary.to_csv(gain_summary_csv, index=False)
    gain_by_selected.to_csv(gain_by_selected_csv, index=False)
    comparison.to_csv(comparison_csv, index=False)
    baselines.to_csv(baselines_csv, index=False)
    repeatability.to_csv(repeatability_csv, index=False)
    normalization_json.write_text(json.dumps(normalization_rows, indent=2), encoding="utf-8")

    figures: list[str] = []
    if not args.skip_plots:
        figures = generate_figures(
            predictions=predictions,
            leaderboard=leaderboard,
            history=history,
            comparison=comparison,
            gain_by_selected=gain_by_selected,
            output_dir=args.figures_dir,
        )

    output_paths = {
        "observer_predictions": as_repo_relative(prediction_meta["path"], ROOT),
        "observer_validation_predictions": as_repo_relative(validation_prediction_meta["path"], ROOT),
        "observer_leaderboard.csv": as_repo_relative(leaderboard_csv, ROOT),
        "observer_validation_leaderboard.csv": as_repo_relative(validation_leaderboard_csv, ROOT),
        "observer_leaderboard.json": as_repo_relative(leaderboard_json, ROOT),
        "observer_branch_metrics.csv": as_repo_relative(branch_csv, ROOT),
        "observer_transition_stable_metrics.csv": as_repo_relative(transition_csv, ROOT),
        "observer_validation_transition_stable_metrics.csv": as_repo_relative(validation_transition_csv, ROOT),
        "observer_validation_test_gap.csv": as_repo_relative(val_test_gap_csv, ROOT),
        "observer_training_history.csv": as_repo_relative(history_csv, ROOT),
        "observer_gain_diagnostics.csv": as_repo_relative(gain_summary_csv, ROOT),
        "observer_gain_by_selected_branch.csv": as_repo_relative(gain_by_selected_csv, ROOT),
        "observer_vs_baselines.csv": as_repo_relative(comparison_csv, ROOT),
        "observer_reference_baselines.csv": as_repo_relative(baselines_csv, ROOT),
        "observer_seed_repeatability.csv": as_repo_relative(repeatability_csv, ROOT),
        "normalization_metadata.json": as_repo_relative(normalization_json, ROOT),
        "checkpoints": as_repo_relative(checkpoint_dir, ROOT),
    }
    write_report(
        path=args.report_path,
        args=args,
        torch_version=torch.__version__,
        device=device,
        cuda_available=bool(torch.cuda.is_available()),
        validation_leaderboard=validation_leaderboard,
        leaderboard=leaderboard,
        branch_metrics=branch_metrics,
        transition_metrics=transition_metrics,
        val_test_gap=val_test_gap,
        history=history,
        baselines=baselines,
        comparison=comparison,
        gain_summary=gain_summary,
        gain_by_selected=gain_by_selected,
        repeatability=repeatability,
        output_paths=output_paths,
        figures=figures,
        metadata=all_metadata,
    )

    best = leaderboard.sort_values("mae").head(1)
    print(
        json.dumps(
            {
                "status": "ok",
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "device": device,
                "runs": int(len(all_metadata)),
                "best": best[["dataset_id", "target_mode", "method", "mae", "dwell_mae", "weighted_energy_error"]].to_dict(
                    orient="records"
                ),
                "outputs": output_paths,
                "figures": [as_repo_relative(path, ROOT) for path in figures],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
