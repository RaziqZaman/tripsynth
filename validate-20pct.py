#!/usr/bin/env python3
"""Validate 80/20 synthetic trips against the held-out 20% household subset."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

from holdout_split import split_rows_by_household
from tabular_io import StringRowWriter, count_rows, get_fieldnames, iter_rows, read_rows, write_rows
from untuple import untuple_households
from validate import (
    NULL_LABEL,
    ColumnScan,
    RunningStats,
    histogram_index,
    js_distance,
    normalized_distribution,
    normalize_value,
    parse_numeric,
    safe_filename,
    scan_dataset,
    stable_bucket,
    tv_distance,
)

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("tupled-survey.parquet"))
    parser.add_argument("--synthetic", type=Path, default=Path("synthetic_trips_80pct.parquet"))
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-households", type=Path, default=Path("heldout_households_20pct.parquet"))
    parser.add_argument("--holdout-trips", type=Path, default=Path("heldout_trips_20pct.parquet"))
    parser.add_argument("--validation-csv", type=Path, default=Path("validation_20pct.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("comparison_histograms_20pct"))
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--hash-buckets", type=int, default=256)
    parser.add_argument("--max-categories-plot", type=int, default=25)
    parser.add_argument("--numeric-threshold", type=float, default=0.98)
    parser.add_argument("--max-columns", type=int, default=None)
    parser.set_defaults(skip_plots=False)
    parser.add_argument("--skip-plots", dest="skip_plots", action="store_true")
    parser.add_argument("--with-plots", dest="skip_plots", action="store_false")
    return parser.parse_args()


def materialize_holdout(args: argparse.Namespace) -> None:
    columns, rows = read_rows(args.input)
    _, holdout_rows = split_rows_by_household(rows, args.holdout_fraction, args.seed)
    with StringRowWriter(args.holdout_households, columns, buffer_size=8192) as writer:
        for row in tqdm(
            holdout_rows,
            desc="Writing held-out households",
            unit="hh",
            total=len(holdout_rows),
        ):
            writer.write(row)
    untuple_households(
        input_path=args.holdout_households,
        output_path=args.holdout_trips,
        sample_output_path=None,
    )


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Missing tupled input: {args.input}")
    if not args.synthetic.exists():
        raise FileNotFoundError(f"Missing synthetic trips file: {args.synthetic}")
    if args.bins < 1:
        raise ValueError("--bins must be >= 1")
    if args.hash_buckets < 2:
        raise ValueError("--hash-buckets must be >= 2")

    materialize_holdout(args)
    if not args.holdout_trips.exists():
        raise FileNotFoundError(f"Failed to materialize holdout trips: {args.holdout_trips}")

    plt = None
    if not args.skip_plots:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ModuleNotFoundError as exc:
            missing = exc.name or "matplotlib"
            raise ModuleNotFoundError(
                f"Missing Python dependency: {missing}. Install plotting support with: pip install matplotlib"
            ) from exc

    synth_cols = set(get_fieldnames(args.synthetic))
    holdout_cols = set(get_fieldnames(args.holdout_trips))
    shared_cols = sorted(synth_cols & holdout_cols)
    if args.max_columns is not None:
        shared_cols = shared_cols[: args.max_columns]
    if not shared_cols:
        raise ValueError("No shared columns found between synthetic trips and holdout trips.")

    synth_total_rows = count_rows(args.synthetic)
    holdout_total_rows = count_rows(args.holdout_trips)
    synth_rows, synth_scan = scan_dataset(
        args.synthetic,
        shared_cols,
        args.max_categories_plot,
        desc="Scanning synthetic",
        total_rows_hint=synth_total_rows,
    )
    holdout_rows, holdout_scan = scan_dataset(
        args.holdout_trips,
        shared_cols,
        args.max_categories_plot,
        desc="Scanning holdout",
        total_rows_hint=holdout_total_rows,
    )

    metadata: Dict[str, dict[str, object]] = {}
    for col in tqdm(shared_cols, desc="Profiling shared columns", unit="col"):
        s = synth_scan[col]
        h = holdout_scan[col]
        s_frac = (s.numeric_count / s.non_empty) if s.non_empty > 0 else 0.0
        h_frac = (h.numeric_count / h.non_empty) if h.non_empty > 0 else 0.0
        is_numeric = (
            s.non_empty > 0
            and h.non_empty > 0
            and s_frac >= args.numeric_threshold
            and h_frac >= args.numeric_threshold
        )

        if is_numeric:
            min_value = min(s.min_value, h.min_value)
            max_value = max(s.max_value, h.max_value)
            if not math.isfinite(min_value) or not math.isfinite(max_value):
                min_value, max_value = 0.0, 1.0
            if max_value <= min_value:
                min_value -= 0.5
                max_value += 0.5
            metadata[col] = {
                "kind": "numeric",
                "min": min_value,
                "max": max_value,
                "s_hist": np.zeros(args.bins, dtype=np.int64),
                "h_hist": np.zeros(args.bins, dtype=np.int64),
                "s_stats": RunningStats(),
                "h_stats": RunningStats(),
            }
        else:
            exact = s.unique_values is not None and h.unique_values is not None
            metadata[col] = {
                "kind": "categorical",
                "mode": "exact" if exact else "hashed",
                "s_count": Counter() if exact else np.zeros(args.hash_buckets, dtype=np.int64),
                "h_count": Counter() if exact else np.zeros(args.hash_buckets, dtype=np.int64),
            }

    numeric_cols = [c for c in shared_cols if metadata[c]["kind"] == "numeric"]
    categorical_exact_cols = [
        c for c in shared_cols if metadata[c]["kind"] == "categorical" and metadata[c]["mode"] == "exact"
    ]
    categorical_hashed_cols = [
        c for c in shared_cols if metadata[c]["kind"] == "categorical" and metadata[c]["mode"] == "hashed"
    ]

    def accumulate(path: Path, side: str, total_rows: int) -> None:
        label = "Accumulating synthetic" if side == "s" else "Accumulating holdout"
        for row in tqdm(iter_rows(path), total=total_rows, desc=label, unit="row"):
            for col in numeric_cols:
                info = metadata[col]
                x = parse_numeric(row.get(col))
                if x is None:
                    continue
                idx = histogram_index(float(x), float(info["min"]), float(info["max"]), args.bins)
                if side == "s":
                    info["s_hist"][idx] += 1
                    info["s_stats"].update(float(x))
                else:
                    info["h_hist"][idx] += 1
                    info["h_stats"].update(float(x))

            for col in categorical_exact_cols:
                info = metadata[col]
                token = normalize_value(row.get(col))
                if token == "":
                    token = NULL_LABEL
                if side == "s":
                    info["s_count"][token] += 1
                else:
                    info["h_count"][token] += 1

            for col in categorical_hashed_cols:
                info = metadata[col]
                token = normalize_value(row.get(col))
                if token == "":
                    token = NULL_LABEL
                bucket = stable_bucket(token, args.hash_buckets)
                if side == "s":
                    info["s_count"][bucket] += 1
                else:
                    info["h_count"][bucket] += 1

    accumulate(args.synthetic, "s", synth_rows)
    accumulate(args.holdout_trips, "h", holdout_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_rows: List[Dict[str, str]] = []

    for col in tqdm(shared_cols, desc="Scoring columns", unit="col"):
        info = metadata[col]
        s_non_empty = synth_scan[col].non_empty
        h_non_empty = holdout_scan[col].non_empty
        s_missing_rate = 1.0 - (s_non_empty / synth_rows if synth_rows > 0 else 0.0)
        h_missing_rate = 1.0 - (h_non_empty / holdout_rows if holdout_rows > 0 else 0.0)

        if info["kind"] == "numeric":
            s_hist = normalized_distribution(info["s_hist"])
            h_hist = normalized_distribution(info["h_hist"])
            metric_js = js_distance(s_hist, h_hist)
            metric_tv = tv_distance(s_hist, h_hist)

            if not args.skip_plots:
                x_min = float(info["min"])
                x_max = float(info["max"])
                edges = np.linspace(x_min, x_max, args.bins + 1)
                centers = 0.5 * (edges[:-1] + edges[1:])
                fig, ax = plt.subplots(figsize=(10, 5.5))
                ax.step(centers, h_hist, where="mid", label="holdout", linewidth=1.5)
                ax.step(centers, s_hist, where="mid", label="synthetic", linewidth=1.5)
                ax.set_title(f"Histogram Match: {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Probability")
                ax.grid(True, alpha=0.2)
                ax.legend()
                fig.tight_layout()
                fig.savefig(args.output_dir / f"{safe_filename(col)}.png", dpi=140)
                plt.close(fig)

            s_stats: RunningStats = info["s_stats"]
            h_stats: RunningStats = info["h_stats"]
            validation_rows.append(
                {
                    "column": col,
                    "column_type": "numeric",
                    "distribution_mode": "histogram",
                    "js_distance": f"{metric_js:.6f}",
                    "tv_distance": f"{metric_tv:.6f}",
                    "synthetic_non_empty": str(s_non_empty),
                    "holdout_non_empty": str(h_non_empty),
                    "synthetic_missing_rate": f"{s_missing_rate:.6f}",
                    "holdout_missing_rate": f"{h_missing_rate:.6f}",
                    "synthetic_mean": f"{s_stats.mean:.6f}",
                    "holdout_mean": f"{h_stats.mean:.6f}",
                    "synthetic_std": f"{s_stats.std:.6f}",
                    "holdout_std": f"{h_stats.std:.6f}",
                }
            )
        else:
            if info["mode"] == "exact":
                s_count: Counter = info["s_count"]
                h_count: Counter = info["h_count"]
                categories = sorted(set(s_count.keys()) | set(h_count.keys()))
                s_arr = np.array([s_count.get(c, 0) for c in categories], dtype=np.float64)
                h_arr = np.array([h_count.get(c, 0) for c in categories], dtype=np.float64)
            else:
                s_arr = info["s_count"].astype(np.float64)
                h_arr = info["h_count"].astype(np.float64)

            p = normalized_distribution(s_arr)
            q = normalized_distribution(h_arr)
            metric_js = js_distance(p, q)
            metric_tv = tv_distance(p, q)

            if not args.skip_plots:
                fig, ax = plt.subplots(figsize=(10, 5.5))
                if info["mode"] == "exact":
                    s_count = info["s_count"]
                    h_count = info["h_count"]
                    top = sorted(
                        set(s_count.keys()) | set(h_count.keys()),
                        key=lambda k: s_count.get(k, 0) + h_count.get(k, 0),
                        reverse=True,
                    )[: args.max_categories_plot]
                    xs = np.arange(len(top), dtype=np.float64)
                    s_vals = normalized_distribution(
                        np.array([s_count.get(k, 0) for k in top], dtype=np.float64)
                    )
                    h_vals = normalized_distribution(
                        np.array([h_count.get(k, 0) for k in top], dtype=np.float64)
                    )
                    width = 0.45
                    ax.bar(xs - width / 2, h_vals, width=width, label="holdout")
                    ax.bar(xs + width / 2, s_vals, width=width, label="synthetic")
                    ax.set_xticks(xs)
                    ax.set_xticklabels(top, rotation=75, ha="right", fontsize=8)
                    ax.set_xlabel("Category")
                else:
                    xs = np.arange(args.hash_buckets, dtype=np.float64)
                    ax.plot(xs, q, label="holdout", linewidth=1.2)
                    ax.plot(xs, p, label="synthetic", linewidth=1.2)
                    ax.set_xlabel("Hashed category bucket")

                ax.set_title(f"Histogram Match: {col}")
                ax.set_ylabel("Probability")
                ax.grid(True, alpha=0.2)
                ax.legend()
                fig.tight_layout()
                fig.savefig(args.output_dir / f"{safe_filename(col)}.png", dpi=140)
                plt.close(fig)

            validation_rows.append(
                {
                    "column": col,
                    "column_type": "categorical",
                    "distribution_mode": str(info["mode"]),
                    "js_distance": f"{metric_js:.6f}",
                    "tv_distance": f"{metric_tv:.6f}",
                    "synthetic_non_empty": str(s_non_empty),
                    "holdout_non_empty": str(h_non_empty),
                    "synthetic_missing_rate": f"{s_missing_rate:.6f}",
                    "holdout_missing_rate": f"{h_missing_rate:.6f}",
                    "synthetic_mean": "",
                    "holdout_mean": "",
                    "synthetic_std": "",
                    "holdout_std": "",
                }
            )

    validation_rows.sort(key=lambda row: (-float(row["js_distance"]), row["column"]))
    write_rows(
        args.validation_csv,
        [
            "column",
            "column_type",
            "distribution_mode",
            "js_distance",
            "tv_distance",
            "synthetic_non_empty",
            "holdout_non_empty",
            "synthetic_missing_rate",
            "holdout_missing_rate",
            "synthetic_mean",
            "holdout_mean",
            "synthetic_std",
            "holdout_std",
        ],
        validation_rows,
    )
    print(f"Wrote {args.validation_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
