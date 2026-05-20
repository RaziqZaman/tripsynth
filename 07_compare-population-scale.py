#!/usr/bin/env python3
"""Compare population-scale synthetic trips against survey trips expanded by weight."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_: object):
        return iterable


REAL_CSV = Path("03x_filled-survey.csv")
SYNTHETIC_CSV = Path("07x_population_scale/synthetic_population_trips.csv")
OUTPUT_DIR = Path("07x_population_scale/comparison")
WEIGHT_COLUMN = "wthhfin"
MISSING_LABEL = "N/A"
OTHER_LABEL = "(other)"
FIFTEEN_MINUTE_BIN_COLUMNS = {"departure_time_min", "reported_travel_time"}
FIFTEEN_MINUTES = 15

NUMERIC_COLUMNS = {
    "departure_time_min",
    "travelers_hh",
    "vehicle_occupancy",
    "distance",
    "reported_travel_time",
    "year",
    "hhsize",
    "numstudents",
    "numdrivers",
    "numworkers",
    "numdisabilities",
    "numvehicle",
    "numvehicle_transponder",
    "numbicycle",
    "hh_income_detailed",
    "tdate_days",
    "hhtrips",
    "age",
    "jobs_count",
    "j1_telecommute_days",
    "td_telecommute_time",
    "td_shop_time",
    "person_tripcount",
    "walk_bike_loop_trips",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-csv", type=Path, default=REAL_CSV)
    parser.add_argument("--synthetic-csv", type=Path, default=SYNTHETIC_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--weight-column", default=WEIGHT_COLUMN)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--top-categories", type=int, default=30)
    return parser.parse_args()


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "value"


def series_labels(column: str, values: pd.Series) -> pd.Series:
    values = values.astype(str)
    missing = values.eq("") | values.str.lower().eq("nan")
    if column in FIFTEEN_MINUTE_BIN_COLUMNS:
        numeric = pd.to_numeric(values, errors="coerce")
        missing = missing | numeric.isna()
        starts = (np.floor(numeric.fillna(0).astype(float) / FIFTEEN_MINUTES) * FIFTEEN_MINUTES).astype(int)
        labels = starts.astype(str) + "-" + (starts + FIFTEEN_MINUTES - 1).astype(str)
        return labels.mask(missing, MISSING_LABEL)
    if column in NUMERIC_COLUMNS:
        numeric = pd.to_numeric(values, errors="coerce")
        missing = missing | numeric.isna()
        labels = np.rint(numeric.fillna(0).astype(float)).astype(int).astype(str)
        return labels.mask(missing, MISSING_LABEL)
    return values.mask(missing, MISSING_LABEL)


def add_weighted_counts(counts: Counter[str], values: pd.Series, weights: pd.Series, column: str) -> None:
    labels = series_labels(column, values)
    grouped = weights.groupby(labels, dropna=False).sum()
    for label, weight in grouped.items():
        counts[str(label)] += float(weight)


def add_unweighted_counts(counts: Counter[str], values: pd.Series, column: str) -> None:
    labels = series_labels(column, values)
    for label, count in labels.value_counts(dropna=False).items():
        counts[str(label)] += float(count)


def distribution_overlap(real_counts: Counter[str], synthetic_counts: Counter[str]) -> float:
    real_total = sum(real_counts.values())
    synthetic_total = sum(synthetic_counts.values())
    if real_total == 0 and synthetic_total == 0:
        return 1.0
    if real_total == 0 or synthetic_total == 0:
        return 0.0
    return sum(
        min(real_counts[value] / real_total, synthetic_counts[value] / synthetic_total)
        for value in set(real_counts) | set(synthetic_counts)
    )


def sort_key(column: str, label: str) -> tuple[int, object]:
    if label == MISSING_LABEL:
        return (1, label)
    if column in FIFTEEN_MINUTE_BIN_COLUMNS and "-" in label:
        return (0, int(label.split("-", 1)[0]))
    if column in NUMERIC_COLUMNS:
        try:
            return (0, int(label))
        except ValueError:
            return (0, label)
    return (0, label)


def chart_labels(
    column: str,
    real_counts: Counter[str],
    synthetic_counts: Counter[str],
    top_categories: int,
) -> tuple[list[str], list[float], list[float]]:
    labels = sorted(set(real_counts) | set(synthetic_counts), key=lambda label: sort_key(column, label))
    if column not in NUMERIC_COLUMNS and len(labels) > top_categories:
        totals = {label: real_counts[label] + synthetic_counts[label] for label in labels}
        selected = [
            label for label, _ in sorted(totals.items(), key=lambda item: item[1], reverse=True)[:top_categories]
        ]
        other_real = sum(real_counts[label] for label in labels if label not in selected)
        other_synthetic = sum(synthetic_counts[label] for label in labels if label not in selected)
        labels = selected + [OTHER_LABEL]
        real_counts = Counter({label: real_counts[label] for label in selected})
        synthetic_counts = Counter({label: synthetic_counts[label] for label in selected})
        real_counts[OTHER_LABEL] = other_real
        synthetic_counts[OTHER_LABEL] = other_synthetic

    real_total = sum(real_counts.values()) or 1.0
    synthetic_total = sum(synthetic_counts.values()) or 1.0
    return (
        labels,
        [real_counts[label] / real_total for label in labels],
        [synthetic_counts[label] / synthetic_total for label in labels],
    )


def plot_column(
    column: str,
    real_counts: Counter[str],
    synthetic_counts: Counter[str],
    out_dir: Path,
    top_categories: int,
) -> None:
    labels, real_props, synthetic_props = chart_labels(
        column, real_counts, synthetic_counts, top_categories
    )
    width = max(12, min(30, 0.35 * len(labels)))
    fig, ax = plt.subplots(figsize=(width, 6))
    x = np.arange(len(labels))
    bar_width = 0.42
    ax.bar(x - bar_width / 2, real_props, bar_width, label="real survey weighted by wthhfin")
    ax.bar(x + bar_width / 2, synthetic_props, bar_width, label="population-scale synthetic")
    ax.set_title(column)
    ax.set_ylabel("proportion")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{safe_filename(column)}.png", dpi=150)
    plt.close(fig)


def read_columns(path: Path) -> list[str]:
    with path.open(newline="") as input_file:
        return list(csv.DictReader(input_file).fieldnames or [])


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    histogram_dir = args.out_dir / "histograms"
    histogram_dir.mkdir(parents=True, exist_ok=True)

    real_columns = read_columns(args.real_csv)
    synthetic_columns = read_columns(args.synthetic_csv)
    common_columns = [
        column for column in real_columns
        if column != args.weight_column and column in set(synthetic_columns)
    ]
    if not common_columns:
        raise SystemExit("No common columns to compare")

    real_counts = {column: Counter() for column in common_columns}
    synthetic_counts = {column: Counter() for column in common_columns}
    real_total_weight = 0.0
    synthetic_rows = 0

    real_usecols = common_columns + [args.weight_column]
    for chunk in tqdm(
        pd.read_csv(args.real_csv, dtype=str, keep_default_na=False, usecols=real_usecols, chunksize=args.chunksize),
        desc="reading weighted real",
        unit="chunk",
    ):
        weights = pd.to_numeric(chunk[args.weight_column], errors="coerce").fillna(0).clip(lower=0)
        real_total_weight += float(weights.sum())
        for column in common_columns:
            add_weighted_counts(real_counts[column], chunk[column], weights, column)

    for chunk in tqdm(
        pd.read_csv(args.synthetic_csv, dtype=str, keep_default_na=False, usecols=common_columns, chunksize=args.chunksize),
        desc="reading synthetic population",
        unit="chunk",
    ):
        synthetic_rows += len(chunk)
        for column in common_columns:
            add_unweighted_counts(synthetic_counts[column], chunk[column], column)

    summary_rows = []
    for column in tqdm(common_columns, desc="writing charts", unit="column"):
        overlap = distribution_overlap(real_counts[column], synthetic_counts[column])
        summary_rows.append(
            {
                "column": column,
                "overlap": f"{overlap:.8f}",
                "total_variation_distance": f"{1.0 - overlap:.8f}",
                "real_weighted_total": f"{sum(real_counts[column].values()):.8f}",
                "synthetic_total": f"{sum(synthetic_counts[column].values()):.8f}",
                "real_distinct_values": len(real_counts[column]),
                "synthetic_distinct_values": len(synthetic_counts[column]),
            }
        )
        plot_column(column, real_counts[column], synthetic_counts[column], histogram_dir, args.top_categories)

    summary_rows.sort(key=lambda row: float(row["overlap"]))
    summary_path = args.out_dir / "population_scale_total_variations.csv"
    with summary_path.open("w", newline="") as output_file:
        fieldnames = [
            "column",
            "overlap",
            "total_variation_distance",
            "real_weighted_total",
            "synthetic_total",
            "real_distinct_values",
            "synthetic_distinct_values",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    with (args.out_dir / "population_scale_summary.csv").open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerow({"metric": "real_weighted_trip_total", "value": f"{real_total_weight:.8f}"})
        writer.writerow({"metric": "synthetic_trip_rows", "value": str(synthetic_rows)})
        writer.writerow({"metric": "common_columns", "value": str(len(common_columns))})

    print(f"real weighted trip total: {real_total_weight:,.2f}")
    print(f"synthetic trip rows: {synthetic_rows:,}")
    print(f"wrote {summary_path}")
    print(f"wrote histograms to {histogram_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
