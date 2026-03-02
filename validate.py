#!/usr/bin/env python3
"""Validate synthetic trips against survey data with per-column histogram matching.

Compares shared columns across:
- synthetic_trips.parquet (default)
- combined-flat-survey.csv (default)

Outputs:
- histogram image per shared column in comparison_histograms/
- validation.csv with standardized distribution differences by column
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from tabular_io import get_fieldnames, iter_rows

DEFAULT_SYNTH = Path("synthetic_trips.parquet")
DEFAULT_SURVEY_CANDIDATES = [
    Path("cobined-flat-survey.csv"),
    Path("combined-flat-survey.csv"),
]
DEFAULT_OUTPUT_DIR = Path("comparison_histograms")
DEFAULT_VALIDATION_CSV = Path("validation.csv")
NULL_LABEL = "<NULL>"


@dataclass
class ColumnScan:
    non_empty: int = 0
    numeric_count: int = 0
    min_value: float = math.inf
    max_value: float = -math.inf
    unique_values: set[str] | None = None


@dataclass
class RunningStats:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.n - 1))


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned or "unnamed_column"


def resolve_default_survey_path() -> Path:
    for candidate in DEFAULT_SURVEY_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_SURVEY_CANDIDATES[0]


def normalize_value(raw: object) -> str:
    if raw is None:
        return ""
    return str(raw).strip()


def parse_numeric(raw: object) -> float | None:
    s = normalize_value(raw)
    if not s:
        return None
    try:
        x = float(s)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def stable_bucket(value: str, n_buckets: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % n_buckets


def scan_dataset(path: Path, columns: List[str], unique_cap: int) -> tuple[int, Dict[str, ColumnScan]]:
    scans: Dict[str, ColumnScan] = {
        c: ColumnScan(unique_values=set()) for c in columns
    }
    total_rows = 0

    for row in iter_rows(path):
        total_rows += 1
        for col in columns:
            raw = row.get(col)
            text = normalize_value(raw)
            scan = scans[col]
            if text:
                scan.non_empty += 1

                x = parse_numeric(raw)
                if x is not None:
                    scan.numeric_count += 1
                    if x < scan.min_value:
                        scan.min_value = x
                    if x > scan.max_value:
                        scan.max_value = x

                if scan.unique_values is not None:
                    scan.unique_values.add(text)
                    if len(scan.unique_values) > unique_cap:
                        scan.unique_values = None

    return total_rows, scans


def normalized_distribution(arr: np.ndarray) -> np.ndarray:
    total = float(arr.sum())
    if total <= 0:
        return np.zeros_like(arr, dtype=np.float64)
    return arr.astype(np.float64) / total


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    mask_p = p > 0
    mask_q = q > 0
    kl_pm = np.sum(np.where(mask_p, p * np.log2(p / np.clip(m, 1e-16, None)), 0.0))
    kl_qm = np.sum(np.where(mask_q, q * np.log2(q / np.clip(m, 1e-16, None)), 0.0))
    js = 0.5 * (kl_pm + kl_qm)
    js = max(0.0, float(js))
    return float(math.sqrt(min(1.0, js)))


def histogram_index(value: float, min_value: float, max_value: float, bins: int) -> int:
    if value <= min_value:
        return 0
    if value >= max_value:
        return bins - 1
    width = (max_value - min_value) / bins
    if width <= 0:
        return 0
    idx = int((value - min_value) / width)
    if idx < 0:
        return 0
    if idx >= bins:
        return bins - 1
    return idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic",
        type=Path,
        default=DEFAULT_SYNTH,
        help=f"Synthetic table path (default: {DEFAULT_SYNTH})",
    )
    parser.add_argument(
        "--survey",
        type=Path,
        default=resolve_default_survey_path(),
        help=(
            "Survey table path (default: first existing of "
            "cobined-flat-survey.csv, combined-flat-survey.csv)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to store histogram PNGs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--validation-csv",
        type=Path,
        default=DEFAULT_VALIDATION_CSV,
        help=f"Validation summary CSV path (default: {DEFAULT_VALIDATION_CSV})",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of bins for numeric histograms (default: 40)",
    )
    parser.add_argument(
        "--hash-buckets",
        type=int,
        default=256,
        help="Buckets for high-cardinality categorical columns (default: 256)",
    )
    parser.add_argument(
        "--max-categories-plot",
        type=int,
        default=25,
        help="Max categories shown for low-cardinality categorical plots (default: 25)",
    )
    parser.add_argument(
        "--numeric-threshold",
        type=float,
        default=0.98,
        help="Min parseable numeric fraction to treat a column as numeric (default: 0.98)",
    )
    parser.add_argument(
        "--max-columns",
        type=int,
        default=None,
        help="Optional cap on number of shared columns validated.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip saving histogram plots.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.synthetic.exists():
        raise FileNotFoundError(f"Missing synthetic file: {args.synthetic}")
    if not args.survey.exists():
        raise FileNotFoundError(f"Missing survey file: {args.survey}")
    if args.bins < 1:
        raise ValueError("--bins must be >= 1")
    if args.hash_buckets < 2:
        raise ValueError("--hash-buckets must be >= 2")

    plt = None
    if not args.skip_plots:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ModuleNotFoundError as exc:
            missing = exc.name or "matplotlib"
            raise ModuleNotFoundError(
                f"Missing Python dependency: {missing}. "
                "Install plotting support with: pip install matplotlib"
            ) from exc

    synth_cols = set(get_fieldnames(args.synthetic))
    survey_cols = set(get_fieldnames(args.survey))
    shared_cols = sorted(synth_cols & survey_cols)
    if args.max_columns is not None:
        shared_cols = shared_cols[: args.max_columns]
    if not shared_cols:
        raise ValueError("No shared columns found between synthetic and survey tables.")

    synth_rows, synth_scan = scan_dataset(args.synthetic, shared_cols, args.max_categories_plot)
    survey_rows, survey_scan = scan_dataset(args.survey, shared_cols, args.max_categories_plot)

    metadata: Dict[str, dict[str, object]] = {}
    for col in shared_cols:
        s = synth_scan[col]
        r = survey_scan[col]

        s_frac = (s.numeric_count / s.non_empty) if s.non_empty > 0 else 0.0
        r_frac = (r.numeric_count / r.non_empty) if r.non_empty > 0 else 0.0
        is_numeric = (
            s.non_empty > 0
            and r.non_empty > 0
            and s_frac >= args.numeric_threshold
            and r_frac >= args.numeric_threshold
        )

        if is_numeric:
            min_value = min(s.min_value, r.min_value)
            max_value = max(s.max_value, r.max_value)
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
                "r_hist": np.zeros(args.bins, dtype=np.int64),
                "s_stats": RunningStats(),
                "r_stats": RunningStats(),
            }
        else:
            exact = s.unique_values is not None and r.unique_values is not None
            metadata[col] = {
                "kind": "categorical",
                "mode": "exact" if exact else "hashed",
                "s_count": Counter() if exact else np.zeros(args.hash_buckets, dtype=np.int64),
                "r_count": Counter() if exact else np.zeros(args.hash_buckets, dtype=np.int64),
            }

    def accumulate(path: Path, side: str) -> None:
        for row in iter_rows(path):
            for col in shared_cols:
                info = metadata[col]
                raw = row.get(col)
                if info["kind"] == "numeric":
                    x = parse_numeric(raw)
                    if x is None:
                        continue
                    idx = histogram_index(float(x), float(info["min"]), float(info["max"]), args.bins)
                    if side == "s":
                        info["s_hist"][idx] += 1
                        info["s_stats"].update(float(x))
                    else:
                        info["r_hist"][idx] += 1
                        info["r_stats"].update(float(x))
                else:
                    token = normalize_value(raw)
                    if token == "":
                        token = NULL_LABEL
                    if info["mode"] == "exact":
                        if side == "s":
                            info["s_count"][token] += 1
                        else:
                            info["r_count"][token] += 1
                    else:
                        b = stable_bucket(token, args.hash_buckets)
                        if side == "s":
                            info["s_count"][b] += 1
                        else:
                            info["r_count"][b] += 1

    accumulate(args.synthetic, "s")
    accumulate(args.survey, "r")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_rows: List[Dict[str, str]] = []

    for col in shared_cols:
        info = metadata[col]
        s_non_empty = synth_scan[col].non_empty
        r_non_empty = survey_scan[col].non_empty
        s_missing_rate = 1.0 - (s_non_empty / synth_rows if synth_rows > 0 else 0.0)
        r_missing_rate = 1.0 - (r_non_empty / survey_rows if survey_rows > 0 else 0.0)

        if info["kind"] == "numeric":
            s_hist = normalized_distribution(info["s_hist"])
            r_hist = normalized_distribution(info["r_hist"])
            metric_js = js_distance(s_hist, r_hist)
            metric_tv = tv_distance(s_hist, r_hist)

            if not args.skip_plots:
                x_min = float(info["min"])
                x_max = float(info["max"])
                edges = np.linspace(x_min, x_max, args.bins + 1)
                centers = 0.5 * (edges[:-1] + edges[1:])

                fig, ax = plt.subplots(figsize=(10, 5.5))
                ax.step(centers, r_hist, where="mid", label="survey", linewidth=1.5)
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
            r_stats: RunningStats = info["r_stats"]
            validation_rows.append(
                {
                    "column": col,
                    "column_type": "numeric",
                    "distribution_mode": "histogram",
                    "js_distance": f"{metric_js:.6f}",
                    "tv_distance": f"{metric_tv:.6f}",
                    "synthetic_non_empty": str(s_non_empty),
                    "survey_non_empty": str(r_non_empty),
                    "synthetic_missing_rate": f"{s_missing_rate:.6f}",
                    "survey_missing_rate": f"{r_missing_rate:.6f}",
                    "synthetic_mean": f"{s_stats.mean:.6f}",
                    "survey_mean": f"{r_stats.mean:.6f}",
                    "synthetic_std": f"{s_stats.std:.6f}",
                    "survey_std": f"{r_stats.std:.6f}",
                }
            )
        else:
            if info["mode"] == "exact":
                s_count: Counter = info["s_count"]
                r_count: Counter = info["r_count"]
                categories = sorted(set(s_count.keys()) | set(r_count.keys()))
                s_arr = np.array([s_count.get(c, 0) for c in categories], dtype=np.float64)
                r_arr = np.array([r_count.get(c, 0) for c in categories], dtype=np.float64)
            else:
                s_arr = info["s_count"].astype(np.float64)
                r_arr = info["r_count"].astype(np.float64)

            p = normalized_distribution(s_arr)
            q = normalized_distribution(r_arr)
            metric_js = js_distance(p, q)
            metric_tv = tv_distance(p, q)

            if not args.skip_plots:
                fig, ax = plt.subplots(figsize=(10, 5.5))
                if info["mode"] == "exact":
                    s_count = info["s_count"]
                    r_count = info["r_count"]
                    top = sorted(
                        set(s_count.keys()) | set(r_count.keys()),
                        key=lambda k: s_count.get(k, 0) + r_count.get(k, 0),
                        reverse=True,
                    )[: args.max_categories_plot]
                    xs = np.arange(len(top), dtype=np.float64)
                    s_vals = np.array([s_count.get(k, 0) for k in top], dtype=np.float64)
                    r_vals = np.array([r_count.get(k, 0) for k in top], dtype=np.float64)
                    s_vals = normalized_distribution(s_vals)
                    r_vals = normalized_distribution(r_vals)
                    width = 0.45
                    ax.bar(xs - width / 2, r_vals, width=width, label="survey")
                    ax.bar(xs + width / 2, s_vals, width=width, label="synthetic")
                    ax.set_xticks(xs)
                    ax.set_xticklabels(top, rotation=75, ha="right", fontsize=8)
                    ax.set_xlabel("Category")
                else:
                    xs = np.arange(args.hash_buckets, dtype=np.float64)
                    ax.plot(xs, q, label="survey", linewidth=1.2)
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
                    "survey_non_empty": str(r_non_empty),
                    "synthetic_missing_rate": f"{s_missing_rate:.6f}",
                    "survey_missing_rate": f"{r_missing_rate:.6f}",
                    "synthetic_mean": "",
                    "survey_mean": "",
                    "synthetic_std": "",
                    "survey_std": "",
                }
            )

    validation_rows.sort(key=lambda r: float(r["js_distance"]), reverse=True)

    out_cols = [
        "column",
        "column_type",
        "distribution_mode",
        "js_distance",
        "tv_distance",
        "synthetic_non_empty",
        "survey_non_empty",
        "synthetic_missing_rate",
        "survey_missing_rate",
        "synthetic_mean",
        "survey_mean",
        "synthetic_std",
        "survey_std",
    ]
    with args.validation_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()
        writer.writerows(validation_rows)

    print(
        f"Validated {len(shared_cols)} shared columns. "
        f"Wrote {args.validation_csv} and {len(shared_cols)} histogram plot(s) to {args.output_dir}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
