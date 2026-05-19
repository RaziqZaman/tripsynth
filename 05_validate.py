#!/usr/bin/env python3
"""Validate synthetic trips against the filled survey with histogram comparisons."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for bare environments.
    def tqdm(iterable, **_: object):
        return iterable


REAL_CSV = Path("03x_filled-survey.csv")
SYNTHETIC_CSV = Path("04x_synthetic-trips.csv")
HISTOGRAM_DIR = Path("05x_histograms")
TOTAL_VARIATIONS_CSV = Path("05x_total-variations.csv")

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

MISSING_LABEL = "N/A"
OTHER_LABEL = "(other)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-csv", type=Path, default=REAL_CSV)
    parser.add_argument("--synthetic-csv", type=Path, default=SYNTHETIC_CSV)
    parser.add_argument("--histogram-dir", type=Path, default=HISTOGRAM_DIR)
    parser.add_argument("--total-variations-csv", type=Path, default=TOTAL_VARIATIONS_CSV)
    parser.add_argument("--numeric-bins", type=int, default=50)
    parser.add_argument("--top-categories", type=int, default=30)
    return parser.parse_args()


def distribution_overlap(real_counts: Counter[str], synthetic_counts: Counter[str]) -> float:
    real_total = sum(real_counts.values())
    synthetic_total = sum(synthetic_counts.values())
    if real_total == 0 and synthetic_total == 0:
        return 1.0
    if real_total == 0 or synthetic_total == 0:
        return 0.0
    values = set(real_counts) | set(synthetic_counts)
    return sum(
        min(real_counts[value] / real_total, synthetic_counts[value] / synthetic_total)
        for value in values
    )


def numeric_counts(
    real: pd.Series,
    synthetic: pd.Series,
    bins: int,
) -> tuple[Counter[str], Counter[str], list[str]]:
    real_numeric = pd.to_numeric(real, errors="coerce")
    synthetic_numeric = pd.to_numeric(synthetic, errors="coerce")
    combined = pd.concat(
        [real_numeric.dropna(), synthetic_numeric.dropna()],
        ignore_index=True,
    )

    real_counts: Counter[str] = Counter()
    synthetic_counts: Counter[str] = Counter()

    if combined.empty:
        real_counts[MISSING_LABEL] = len(real)
        synthetic_counts[MISSING_LABEL] = len(synthetic)
        return real_counts, synthetic_counts, [MISSING_LABEL]

    min_value = float(combined.min())
    max_value = float(combined.max())
    if math.isclose(min_value, max_value):
        edges = np.array([min_value - 0.5, max_value + 0.5])
    else:
        edges = np.linspace(min_value, max_value, bins + 1)

    labels = [
        f"{edges[index]:.3g}-{edges[index + 1]:.3g}"
        for index in range(len(edges) - 1)
    ]

    real_hist, _ = np.histogram(real_numeric.dropna().to_numpy(), bins=edges)
    synthetic_hist, _ = np.histogram(synthetic_numeric.dropna().to_numpy(), bins=edges)

    real_counts.update(dict(zip(labels, real_hist.tolist())))
    synthetic_counts.update(dict(zip(labels, synthetic_hist.tolist())))

    real_missing = int(real_numeric.isna().sum())
    synthetic_missing = int(synthetic_numeric.isna().sum())
    if real_missing or synthetic_missing:
        real_counts[MISSING_LABEL] = real_missing
        synthetic_counts[MISSING_LABEL] = synthetic_missing
        labels.append(MISSING_LABEL)

    return real_counts, synthetic_counts, labels


def categorical_counts(
    real: pd.Series,
    synthetic: pd.Series,
) -> tuple[Counter[str], Counter[str], list[str]]:
    real_counts = Counter(real.astype(str))
    synthetic_counts = Counter(synthetic.astype(str))
    labels = sorted(set(real_counts) | set(synthetic_counts))
    return real_counts, synthetic_counts, labels


def chart_labels(
    real_counts: Counter[str],
    synthetic_counts: Counter[str],
    labels: list[str],
    top_categories: int,
    categorical: bool,
) -> tuple[list[str], list[float], list[float]]:
    if categorical and len(labels) > top_categories:
        totals = {
            label: real_counts[label] + synthetic_counts[label]
            for label in labels
        }
        selected = [
            label
            for label, _ in sorted(totals.items(), key=lambda item: item[1], reverse=True)[
                :top_categories
            ]
        ]
        other_real = sum(real_counts[label] for label in labels if label not in selected)
        other_synthetic = sum(
            synthetic_counts[label] for label in labels if label not in selected
        )
        labels = selected + [OTHER_LABEL]
        real_counts = Counter({label: real_counts[label] for label in selected})
        synthetic_counts = Counter(
            {label: synthetic_counts[label] for label in selected}
        )
        real_counts[OTHER_LABEL] = other_real
        synthetic_counts[OTHER_LABEL] = other_synthetic

    real_total = sum(real_counts.values()) or 1
    synthetic_total = sum(synthetic_counts.values()) or 1
    real_props = [real_counts[label] / real_total for label in labels]
    synthetic_props = [synthetic_counts[label] / synthetic_total for label in labels]
    return labels, real_props, synthetic_props


def plot_comparison(
    column: str,
    labels: list[str],
    real_props: list[float],
    synthetic_props: list[float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    width = max(12, min(28, 0.35 * len(labels)))
    fig, ax = plt.subplots(figsize=(width, 6))
    x = np.arange(len(labels))
    bar_width = 0.42
    ax.bar(x - bar_width / 2, real_props, bar_width, label="03x filled survey")
    ax.bar(x + bar_width / 2, synthetic_props, bar_width, label="04x synthetic trips")
    ax.set_title(column)
    ax.set_ylabel("proportion")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{column}.png", dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    real = pd.read_csv(args.real_csv, dtype=str, keep_default_na=False)
    synthetic = pd.read_csv(args.synthetic_csv, dtype=str, keep_default_na=False)

    common_columns = [column for column in real.columns if column in synthetic.columns]
    missing_from_synthetic = sorted(set(real.columns) - set(synthetic.columns))
    extra_in_synthetic = sorted(set(synthetic.columns) - set(real.columns))

    rows = []
    for column in tqdm(common_columns, desc="validating columns", unit="column"):
        is_numeric = column in NUMERIC_COLUMNS
        if is_numeric:
            real_counts, synthetic_counts, labels = numeric_counts(
                real[column],
                synthetic[column],
                args.numeric_bins,
            )
        else:
            real_counts, synthetic_counts, labels = categorical_counts(
                real[column],
                synthetic[column],
            )

        overlap = distribution_overlap(real_counts, synthetic_counts)
        charted_labels, real_props, synthetic_props = chart_labels(
            real_counts,
            synthetic_counts,
            labels,
            args.top_categories,
            categorical=not is_numeric,
        )
        plot_comparison(
            column,
            charted_labels,
            real_props,
            synthetic_props,
            args.histogram_dir,
        )
        rows.append(
            {
                "column": column,
                "kind": "numeric" if is_numeric else "categorical",
                "real_unique_values": len(real_counts),
                "synthetic_unique_values": len(synthetic_counts),
                "histogram_overlap": f"{overlap:.8f}",
                "total_variation_distance": f"{1 - overlap:.8f}",
            }
        )

    with args.total_variations_csv.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "column",
                "kind",
                "real_unique_values",
                "synthetic_unique_values",
                "histogram_overlap",
                "total_variation_distance",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.total_variations_csv}")
    print(f"wrote histogram charts to {args.histogram_dir}")
    print(f"compared columns: {len(common_columns)}")
    if missing_from_synthetic:
        print(f"missing from synthetic: {', '.join(missing_from_synthetic)}")
    if extra_in_synthetic:
        print(f"extra in synthetic: {', '.join(extra_in_synthetic)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
