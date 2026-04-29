#!/usr/bin/env python3
"""Run 80/20 method synthesis, baseline comparison, and summary figures."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--validation-interval", type=int, default=4)
    parser.add_argument("--sample-rows", type=int, default=55440)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--latent-dim", type=int, default=1792)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--tv-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--tv-every-n-batches",
        type=int,
        default=1,
        help="Compute TV every N batches; 0 computes it once per epoch at the middle batch.",
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    py = sys.executable

    plot_flag = "--skip-plots" if args.skip_plots else "--with-plots"
    compile_flags = [] if args.compile else ["--no-compile"]

    run(
        [
            py,
            "synthesize-8020.py",
            "--epochs",
            str(args.epochs),
            "--validation-interval",
            str(args.validation_interval),
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
            str(args.reconstruction_weight),
            "--contrastive-weight",
            str(args.contrastive_weight),
            "--tv-loss-weight",
            str(args.tv_loss_weight),
            "--tv-every-n-batches",
            str(args.tv_every_n_batches),
            plot_flag,
            *compile_flags,
        ],
        cwd=repo_root,
    )

    run(
        [
            py,
            "random_baseline_8020.py",
            "--sample-rows",
            str(args.sample_rows),
            plot_flag,
        ],
        cwd=repo_root,
    )

    run([py, "visualize_method_vs_baseline.py"], cwd=repo_root)
    print("80/20 comparison pipeline completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
