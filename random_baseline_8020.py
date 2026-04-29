#!/usr/bin/env python3
"""Random-with-replacement 80/20 baseline.

The baseline samples tupled household rows with replacement from the 80% training
split, untuples them into trips, validates against the held-out 20%, and compares
the validation metrics to the existing 80/20 model run.
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from holdout_split import split_rows_by_household
from tabular_io import StringRowWriter, read_rows
from untuple import untuple_households
from validate import is_id_column

try:
    from tqdm.auto import tqdm
except Exception:

    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("tupled-survey.parquet"))
    parser.add_argument("--household-output", type=Path, default=Path("baseline_synthesized_household_trips_80pct.parquet"))
    parser.add_argument("--trip-output", type=Path, default=Path("baseline_synthetic_trips_80pct.parquet"))
    parser.add_argument("--sample-trip-output", type=Path, default=Path("sample_baseline_synthetic_trips_80pct.csv"))
    parser.add_argument("--holdout-households", type=Path, default=Path("heldout_households_20pct.parquet"))
    parser.add_argument("--holdout-trips", type=Path, default=Path("heldout_trips_20pct.parquet"))
    parser.add_argument("--validation-csv", type=Path, default=Path("validation_baseline_20pct.csv"))
    parser.add_argument("--method-validation-csv", type=Path, default=Path("validation_20pct.csv"))
    parser.add_argument("--comparison-csv", type=Path, default=Path("validation_method_vs_baseline_20pct.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("comparison_histograms_baseline_20pct"))
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--sample-rows", type=int, default=10080)
    parser.add_argument("--sample-households", dest="sample_rows", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--hash-buckets", type=int, default=256)
    parser.add_argument("--max-categories-plot", type=int, default=25)
    parser.add_argument("--numeric-threshold", type=float, default=0.98)
    parser.add_argument("--max-columns", type=int, default=None)
    parser.set_defaults(skip_plots=False)
    parser.add_argument("--skip-plots", dest="skip_plots", action="store_true")
    parser.add_argument("--with-plots", dest="skip_plots", action="store_false")
    return parser.parse_args()


def synthesize_baseline_households(args: argparse.Namespace) -> None:
    columns, rows = read_rows(args.input)
    train_rows, holdout_rows = split_rows_by_household(rows, args.holdout_fraction, args.seed)
    if not train_rows:
        raise ValueError("Training split is empty; cannot sample baseline households.")
    if args.sample_rows < 1:
        raise ValueError("--sample-rows must be >= 1")

    rng = random.Random(args.seed)
    with StringRowWriter(args.household_output, columns, buffer_size=8192) as writer:
        for _ in tqdm(range(args.sample_rows), desc="Sampling baseline households", unit="hh"):
            sampled = dict(rng.choice(train_rows))
            for col in columns:
                if is_id_column(col):
                    sampled[col] = ""
            if "wthhfin" in sampled:
                sampled["wthhfin"] = "1"
            writer.write(sampled)

    print(
        f"train_households={len(train_rows)} holdout_households={len(holdout_rows)} "
        f"sampled_baseline_households={args.sample_rows}"
    )
    print(f"Wrote {args.household_output}")


def validate_baseline(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "validate-20pct.py",
        "--input",
        str(args.input),
        "--synthetic",
        str(args.trip_output),
        "--holdout-fraction",
        str(args.holdout_fraction),
        "--seed",
        str(args.seed),
        "--holdout-households",
        str(args.holdout_households),
        "--holdout-trips",
        str(args.holdout_trips),
        "--validation-csv",
        str(args.validation_csv),
        "--output-dir",
        str(args.output_dir),
        "--bins",
        str(args.bins),
        "--hash-buckets",
        str(args.hash_buckets),
        "--max-categories-plot",
        str(args.max_categories_plot),
        "--numeric-threshold",
        str(args.numeric_threshold),
    ]
    if args.max_columns is not None:
        cmd.extend(["--max-columns", str(args.max_columns)])
    if args.skip_plots:
        cmd.append("--skip-plots")
    else:
        cmd.append("--with-plots")

    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def read_validation(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return {row["column"]: row for row in csv.DictReader(f)}


def write_metric_comparison(args: argparse.Namespace) -> None:
    if not args.method_validation_csv.exists():
        print(f"Skipping method comparison; missing {args.method_validation_csv}")
        return

    method = read_validation(args.method_validation_csv)
    baseline = read_validation(args.validation_csv)
    shared_cols = sorted(set(method) & set(baseline))
    fieldnames = [
        "column",
        "column_type",
        "distribution_mode",
        "method_js_distance",
        "baseline_js_distance",
        "js_delta_method_minus_baseline",
        "js_improvement_vs_baseline",
        "method_tv_distance",
        "baseline_tv_distance",
        "tv_delta_method_minus_baseline",
        "tv_improvement_vs_baseline",
    ]
    rows: List[Dict[str, str]] = []
    for col in shared_cols:
        m = method[col]
        b = baseline[col]
        method_js = float(m["js_distance"])
        baseline_js = float(b["js_distance"])
        method_tv = float(m["tv_distance"])
        baseline_tv = float(b["tv_distance"])
        rows.append(
            {
                "column": col,
                "column_type": m["column_type"],
                "distribution_mode": m["distribution_mode"],
                "method_js_distance": f"{method_js:.6f}",
                "baseline_js_distance": f"{baseline_js:.6f}",
                "js_delta_method_minus_baseline": f"{method_js - baseline_js:.6f}",
                "js_improvement_vs_baseline": f"{baseline_js - method_js:.6f}",
                "method_tv_distance": f"{method_tv:.6f}",
                "baseline_tv_distance": f"{baseline_tv:.6f}",
                "tv_delta_method_minus_baseline": f"{method_tv - baseline_tv:.6f}",
                "tv_improvement_vs_baseline": f"{baseline_tv - method_tv:.6f}",
            }
        )
    rows.sort(key=lambda row: float(row["js_improvement_vs_baseline"]))

    with args.comparison_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        mean_js_delta = sum(float(r["js_delta_method_minus_baseline"]) for r in rows) / len(rows)
        wins = sum(1 for r in rows if float(r["js_delta_method_minus_baseline"]) < 0)
        print(
            f"Wrote {args.comparison_csv} "
            f"(method lower JS on {wins}/{len(rows)} columns; mean JS delta={mean_js_delta:.6f})"
        )
    else:
        print(f"Wrote {args.comparison_csv} (no shared validation columns)")


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Missing tupled input: {args.input}")

    synthesize_baseline_households(args)
    untuple_households(
        input_path=args.household_output,
        output_path=args.trip_output,
        sample_output_path=args.sample_trip_output,
        sample_rows=args.sample_rows,
        seed=args.seed,
    )
    print(f"Wrote {args.trip_output}")

    validate_baseline(args)
    write_metric_comparison(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
