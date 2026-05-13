#!/usr/bin/env python3
"""Sweep reconstruction/contrastive loss weights for the 80/20 experiment."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class RatioRun:
    label: str
    reconstruction_weight: float
    contrastive_weight: float


DEFAULT_RATIOS = [
    RatioRun("recon_1_contrastive_0", 1.0, 0.0),
    RatioRun("recon_2over3_contrastive_1over3", 2.0 / 3.0, 1.0 / 3.0),
    RatioRun("recon_1over2_contrastive_1over2", 0.5, 0.5),
    RatioRun("recon_1over3_contrastive_2over3", 1.0 / 3.0, 2.0 / 3.0),
    RatioRun("recon_0_contrastive_1", 0.0, 1.0),
]


def run(cmd: Sequence[str], cwd: Path) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(list(cmd), cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("loss_weight_sweep_20pct"))
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--validation-interval", type=int, default=24)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=0)
    parser.add_argument("--sample-rows", type=int, default=55440)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--latent-dim", type=int, default=1792)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--tv-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--tv-every-n-batches",
        type=int,
        default=1,
        help="Compute TV every N batches; 0 computes it once per epoch at the middle batch.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--only",
        action="append",
        choices=[run.label for run in DEFAULT_RATIOS],
        help="Run only a named ratio. Can be repeated.",
    )
    return parser.parse_args()


def selected_ratios(args: argparse.Namespace) -> list[RatioRun]:
    if not args.only:
        return list(DEFAULT_RATIOS)
    wanted = set(args.only)
    return [ratio for ratio in DEFAULT_RATIOS if ratio.label in wanted]


def read_validation_summary(path: Path) -> tuple[float, float, int]:
    if not path.exists():
        return float("nan"), float("nan"), 0

    tv_values: list[float] = []
    js_values: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                tv_values.append(float(row.get("tv_distance", "nan")))
            except ValueError:
                pass
            try:
                js_values.append(float(row.get("js_distance", "nan")))
            except ValueError:
                pass

    def mean(values: list[float]) -> float:
        finite = [value for value in values if value == value]
        return sum(finite) / len(finite) if finite else float("nan")

    return mean(tv_values), mean(js_values), max(len(tv_values), len(js_values))


def write_sweep_summary(output_root: Path, ratios: Sequence[RatioRun]) -> Path:
    summary_csv = output_root / "sweep_summary.csv"
    fieldnames = [
        "label",
        "reconstruction_weight",
        "contrastive_weight",
        "mean_tv_distance",
        "mean_js_distance",
        "validation_columns",
        "loss_csv",
        "loss_curve_png",
        "validation_csv",
        "histogram_dir",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ratio in ratios:
            run_dir = output_root / ratio.label
            validation_csv = run_dir / "validation_20pct.csv"
            mean_tv, mean_js, n_cols = read_validation_summary(validation_csv)
            writer.writerow(
                {
                    "label": ratio.label,
                    "reconstruction_weight": f"{ratio.reconstruction_weight:.8f}",
                    "contrastive_weight": f"{ratio.contrastive_weight:.8f}",
                    "mean_tv_distance": f"{mean_tv:.8f}",
                    "mean_js_distance": f"{mean_js:.8f}",
                    "validation_columns": str(n_cols),
                    "loss_csv": str(run_dir / "training_loss_20pct.csv"),
                    "loss_curve_png": str(run_dir / "training_loss_curves_20pct.png"),
                    "validation_csv": str(validation_csv),
                    "histogram_dir": str(run_dir / "comparison_histograms_20pct"),
                }
            )
    return summary_csv


def save_aggregate_plots(output_root: Path, ratios: Sequence[RatioRun]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        return

    loss_fig, loss_axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    validation_fig, validation_axes = plt.subplots(1, 2, figsize=(12, 5))
    plotted_validation = False

    for ratio in ratios:
        run_dir = output_root / ratio.label
        loss_csv = run_dir / "training_loss_20pct.csv"
        if loss_csv.exists():
            epochs: list[int] = []
            total_losses: list[float] = []
            recon_losses: list[float] = []
            contrastive_losses: list[float] = []
            with loss_csv.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    epochs.append(int(row["epoch"]))
                    total_losses.append(float(row["total_loss"]))
                    recon_losses.append(float(row["reconstruction_loss"]))
                    contrastive_losses.append(float(row["contrastive_loss"]))

            short_label = f"{ratio.reconstruction_weight:.3g}:{ratio.contrastive_weight:.3g}"
            loss_axes[0].plot(epochs, total_losses, linewidth=1.5, label=short_label)
            loss_axes[1].plot(epochs, recon_losses, linewidth=1.4, label=f"{short_label} recon")
            loss_axes[1].plot(
                epochs,
                contrastive_losses,
                linewidth=1.2,
                linestyle="--",
                label=f"{short_label} ctr",
            )

        validation_csv = run_dir / "validation_20pct.csv"
        if validation_csv.exists():
            tv_values: list[float] = []
            js_values: list[float] = []
            with validation_csv.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    try:
                        tv_values.append(float(row["tv_distance"]))
                        js_values.append(float(row["js_distance"]))
                    except (KeyError, ValueError):
                        continue
            if tv_values and js_values:
                plotted_validation = True
                short_label = f"{ratio.reconstruction_weight:.3g}:{ratio.contrastive_weight:.3g}"
                validation_axes[0].hist(tv_values, bins=25, histtype="step", linewidth=1.5, label=short_label)
                validation_axes[1].hist(js_values, bins=25, histtype="step", linewidth=1.5, label=short_label)

    loss_axes[0].set_title("Total Training Loss by Weight Ratio")
    loss_axes[0].set_ylabel("Total loss")
    loss_axes[0].grid(True, alpha=0.2)
    loss_axes[0].legend(title="recon:contrastive")
    loss_axes[1].set_title("Reconstruction and Contrastive Loss Components")
    loss_axes[1].set_xlabel("Epoch")
    loss_axes[1].set_ylabel("Component loss")
    loss_axes[1].grid(True, alpha=0.2)
    loss_axes[1].legend(ncol=2, fontsize=8)
    loss_fig.tight_layout()
    loss_fig.savefig(output_root / "sweep_loss_curves.png", dpi=160)
    plt.close(loss_fig)

    if plotted_validation:
        validation_axes[0].set_title("Validation TV Distance Distribution")
        validation_axes[0].set_xlabel("TV distance")
        validation_axes[0].set_ylabel("Columns")
        validation_axes[1].set_title("Validation JS Distance Distribution")
        validation_axes[1].set_xlabel("JS distance")
        validation_axes[1].set_ylabel("Columns")
        for ax in validation_axes:
            ax.grid(True, alpha=0.2)
            ax.legend(title="recon:contrastive")
        validation_fig.tight_layout()
        validation_fig.savefig(output_root / "sweep_validation_metric_histograms.png", dpi=160)
    plt.close(validation_fig)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    plot_flag = "--skip-plots" if args.skip_plots else "--with-plots"
    compile_flags = [] if args.compile else ["--no-compile"]
    ratios = selected_ratios(args)

    for ratio in ratios:
        run_dir = output_root / ratio.label
        run_dir.mkdir(parents=True, exist_ok=True)
        run(
            [
                py,
                "synthesize-8020.py",
                "--household-output",
                str(run_dir / "synthesized-household-trips-80pct.parquet"),
                "--trip-output",
                str(run_dir / "synthetic_trips_80pct.parquet"),
                "--sample-trip-output",
                str(run_dir / "sample_synthetic_trips_80pct.csv"),
                "--holdout-households",
                str(run_dir / "heldout_households_20pct.parquet"),
                "--holdout-trips",
                str(run_dir / "heldout_trips_20pct.parquet"),
                "--validation-csv",
                str(run_dir / "validation_20pct.csv"),
                "--validation-history-csv",
                str(run_dir / "validation_history_20pct.csv"),
                "--validation-checkpoint-dir",
                str(run_dir / "validation_checkpoints_20pct"),
                "--training-loss-csv",
                str(run_dir / "training_loss_20pct.csv"),
                "--training-loss-plot",
                str(run_dir / "training_loss_curves_20pct.png"),
                "--training-checkpoint",
                str(run_dir / "training_checkpoint_20pct.pt"),
                "--output-dir",
                str(run_dir / "comparison_histograms_20pct"),
                "--epochs",
                str(args.epochs),
                "--validation-interval",
                str(args.validation_interval),
                "--checkpoint-every-epochs",
                str(args.checkpoint_every_epochs),
                "--sample-rows",
                str(args.sample_rows),
                "--batch-size",
                str(args.batch_size),
                "--sample-batch-size",
                str(args.sample_batch_size),
                "--hidden-dim",
                str(args.hidden_dim),
                "--latent-dim",
                str(args.latent_dim),
                "--hidden-layers",
                str(args.hidden_layers),
                "--reconstruction-weight",
                str(ratio.reconstruction_weight),
                "--contrastive-weight",
                str(ratio.contrastive_weight),
                "--tv-loss-weight",
                str(args.tv_loss_weight),
                "--tv-every-n-batches",
                str(args.tv_every_n_batches),
                "--seed",
                str(args.seed),
                plot_flag,
                *compile_flags,
            ],
            cwd=repo_root,
        )

    summary_csv = write_sweep_summary(output_root, ratios)
    save_aggregate_plots(output_root, ratios)
    print(f"Wrote {summary_csv}", flush=True)
    print(f"Wrote {output_root / 'sweep_loss_curves.png'}", flush=True)
    print(f"Wrote {output_root / 'sweep_validation_metric_histograms.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
