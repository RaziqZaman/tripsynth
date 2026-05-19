#!/usr/bin/env python3
"""Validate synthetic trips against the filled survey with histogram comparisons."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from itertools import combinations
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
CROSS_MARGINAL_DIR = Path("05x_cross-marginals")

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
FIFTEEN_MINUTE_BIN_COLUMNS = {"departure_time_min", "reported_travel_time"}
FIFTEEN_MINUTES = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-csv", type=Path, default=REAL_CSV)
    parser.add_argument("--synthetic-csv", type=Path, default=SYNTHETIC_CSV)
    parser.add_argument("--histogram-dir", type=Path, default=HISTOGRAM_DIR)
    parser.add_argument("--total-variations-csv", type=Path, default=TOTAL_VARIATIONS_CSV)
    parser.add_argument("--cross-marginal-dir", type=Path, default=CROSS_MARGINAL_DIR)
    parser.add_argument("--numeric-bins", type=int, default=50)
    parser.add_argument("--top-categories", type=int, default=30)
    parser.add_argument(
        "--cross-marginal-plots",
        type=int,
        default=40,
        help="number of worst-overlap pairwise cross-marginals to chart",
    )
    parser.add_argument(
        "--cross-marginal-top-values",
        type=int,
        default=25,
        help="number of joint values to show in each cross-marginal chart",
    )
    return parser.parse_args()


def distribution_overlap(real_counts: Counter, synthetic_counts: Counter) -> float:
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


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "value"


def numeric_counts(
    column: str,
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

    if column in FIFTEEN_MINUTE_BIN_COLUMNS:
        real_binned = real_numeric.dropna().apply(fifteen_minute_bin_label)
        synthetic_binned = synthetic_numeric.dropna().apply(fifteen_minute_bin_label)
        real_counts.update(Counter(real_binned))
        synthetic_counts.update(Counter(synthetic_binned))
        labels = sorted(
            set(real_counts) | set(synthetic_counts),
            key=fifteen_minute_bin_start,
        )
    elif is_integer_like(combined):
        real_integer = real_numeric.dropna().round().astype(int).astype(str)
        synthetic_integer = synthetic_numeric.dropna().round().astype(int).astype(str)
        real_counts.update(Counter(real_integer))
        synthetic_counts.update(Counter(synthetic_integer))
        labels = sorted(
            set(real_counts) | set(synthetic_counts),
            key=lambda value: int(value),
        )
    else:
        min_value = float(combined.min())
        max_value = float(combined.max())
        if math.isclose(min_value, max_value):
            edges = np.array([min_value - 0.5, max_value + 0.5])
        else:
            edges = np.linspace(min_value, max_value, bins + 1)

        labels = [
            integer_bin_label(edges[index], edges[index + 1])
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


def is_integer_like(values: pd.Series) -> bool:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if numeric.size == 0:
        return False
    return bool(np.all(np.isclose(numeric, np.round(numeric))))


def integer_bin_label(left: float, right: float) -> str:
    left_text = str(int(round(left)))
    right_text = str(int(round(right)))
    if left_text == right_text:
        return left_text
    return f"{left_text}-{right_text}"


def numeric_range_label(left: float, right: float) -> str:
    if math.isclose(left, right):
        return f"{left:.6g}"
    return f"{left:.6g}-{right:.6g}"


def fifteen_minute_bin_label(value: float) -> str:
    start = int(math.floor(float(value) / FIFTEEN_MINUTES) * FIFTEEN_MINUTES)
    end = start + FIFTEEN_MINUTES - 1
    return f"{start}-{end}"


def fifteen_minute_bin_start(label: str) -> int:
    return int(label.split("-", 1)[0])


def bucketed_numeric_columns(
    column: str,
    real: pd.Series,
    synthetic: pd.Series,
    bins: int,
) -> tuple[pd.Series, pd.Series]:
    real_numeric = pd.to_numeric(real, errors="coerce")
    synthetic_numeric = pd.to_numeric(synthetic, errors="coerce")
    combined = pd.concat(
        [real_numeric.dropna(), synthetic_numeric.dropna()],
        ignore_index=True,
    )

    if combined.empty:
        return (
            pd.Series([MISSING_LABEL] * len(real), index=real.index),
            pd.Series([MISSING_LABEL] * len(synthetic), index=synthetic.index),
        )

    if column in FIFTEEN_MINUTE_BIN_COLUMNS:
        real_bucketed = real_numeric.apply(
            lambda value: fifteen_minute_bin_label(value) if pd.notna(value) else MISSING_LABEL
        )
        synthetic_bucketed = synthetic_numeric.apply(
            lambda value: fifteen_minute_bin_label(value) if pd.notna(value) else MISSING_LABEL
        )
    elif is_integer_like(combined):
        real_bucketed = real_numeric.round().astype("Int64").astype(str)
        synthetic_bucketed = synthetic_numeric.round().astype("Int64").astype(str)
    else:
        min_value = float(combined.min())
        max_value = float(combined.max())
        if math.isclose(min_value, max_value):
            edges = np.array([min_value - 0.5, max_value + 0.5])
        else:
            edges = np.linspace(min_value, max_value, bins + 1)
        labels = [
            numeric_range_label(edges[index], edges[index + 1])
            for index in range(len(edges) - 1)
        ]
        real_bucketed = pd.cut(
            real_numeric,
            bins=edges,
            labels=labels,
            include_lowest=True,
        ).astype("string")
        synthetic_bucketed = pd.cut(
            synthetic_numeric,
            bins=edges,
            labels=labels,
            include_lowest=True,
        ).astype("string")

    real_bucketed = real_bucketed.mask(real_numeric.isna(), MISSING_LABEL).astype(str)
    synthetic_bucketed = synthetic_bucketed.mask(
        synthetic_numeric.isna(),
        MISSING_LABEL,
    ).astype(str)
    return real_bucketed, synthetic_bucketed


def bucketed_columns(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    common_columns: list[str],
    numeric_bins: int,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    real_bucketed: dict[str, pd.Series] = {}
    synthetic_bucketed: dict[str, pd.Series] = {}
    for column in common_columns:
        if column in NUMERIC_COLUMNS:
            real_column, synthetic_column = bucketed_numeric_columns(
                column,
                real[column],
                synthetic[column],
                numeric_bins,
            )
        else:
            real_column = real[column].astype(str)
            synthetic_column = synthetic[column].astype(str)
        real_bucketed[column] = real_column
        synthetic_bucketed[column] = synthetic_column
    return real_bucketed, synthetic_bucketed


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


def joint_counts(left: pd.Series, right: pd.Series) -> Counter[tuple[str, str]]:
    return Counter(zip(left.astype(str), right.astype(str), strict=True))


def select_axis_labels(
    real_counts: Counter[tuple[str, str]],
    synthetic_counts: Counter[tuple[str, str]],
    axis: int,
    top_values: int,
) -> list[str]:
    counts: Counter[str] = Counter()
    for pair, count in real_counts.items():
        counts[pair[axis]] += count
    for pair, count in synthetic_counts.items():
        counts[pair[axis]] += count
    return [
        value
        for value, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)[
            :top_values
        ]
    ]


def compact_pair(value: tuple[str, str], row_labels: set[str], column_labels: set[str]) -> tuple[str, str]:
    row, column = value
    if row not in row_labels:
        row = OTHER_LABEL
    if column not in column_labels:
        column = OTHER_LABEL
    return row, column


def joint_proportion_matrix(
    counts: Counter[tuple[str, str]],
    row_labels: list[str],
    column_labels: list[str],
) -> np.ndarray:
    rows = row_labels + [OTHER_LABEL]
    columns = column_labels + [OTHER_LABEL]
    row_index = {label: index for index, label in enumerate(rows)}
    column_index = {label: index for index, label in enumerate(columns)}
    selected_rows = set(row_labels)
    selected_columns = set(column_labels)
    matrix = np.zeros((len(rows), len(columns)), dtype=float)
    total = sum(counts.values()) or 1

    for pair, count in counts.items():
        row, column = compact_pair(pair, selected_rows, selected_columns)
        matrix[row_index[row], column_index[column]] += count / total
    return matrix


def plot_cross_marginal(
    column_a: str,
    column_b: str,
    real_counts: Counter[tuple[str, str]],
    synthetic_counts: Counter[tuple[str, str]],
    output_dir: Path,
    top_values: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row_labels = select_axis_labels(real_counts, synthetic_counts, 0, top_values)
    column_labels = select_axis_labels(real_counts, synthetic_counts, 1, top_values)
    rows = row_labels + [OTHER_LABEL]
    columns = column_labels + [OTHER_LABEL]

    real_matrix = joint_proportion_matrix(real_counts, row_labels, column_labels)
    synthetic_matrix = joint_proportion_matrix(synthetic_counts, row_labels, column_labels)
    difference_matrix = synthetic_matrix - real_matrix
    shared_max = max(float(real_matrix.max()), float(synthetic_matrix.max()), 1e-12)
    difference_max = max(float(np.abs(difference_matrix).max()), 1e-12)

    fig_width = max(13, min(32, 0.45 * len(columns) * 3))
    fig_height = max(8, min(28, 0.38 * len(rows)))
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), sharey=True)

    panels = [
        (axes[0], real_matrix, "03x filled survey", "viridis", 0.0, shared_max),
        (axes[1], synthetic_matrix, "04x synthetic trips", "viridis", 0.0, shared_max),
        (axes[2], difference_matrix, "synthetic - real", "coolwarm", -difference_max, difference_max),
    ]
    for ax, matrix, title, cmap, vmin, vmax in panels:
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(columns)))
        ax.set_xticklabels(columns, rotation=75, ha="right")
        ax.grid(False)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    axes[0].set_yticks(np.arange(len(rows)))
    axes[0].set_yticklabels(rows)
    axes[0].set_ylabel(column_a)
    for ax in axes:
        ax.set_xlabel(column_b)

    fig.suptitle(f"{column_a} x {column_b}")
    fig.tight_layout()
    filename = f"{safe_filename(column_a)}__x__{safe_filename(column_b)}.png"
    fig.savefig(output_dir / filename, dpi=150)
    plt.close(fig)


def clear_cross_marginal_pair_plots(output_dir: Path) -> None:
    for path in output_dir.glob("*.png"):
        if path.name != "cross_marginal_overlap_heatmap.png":
            path.unlink()


def plot_cross_marginal_heatmap(
    matrix: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    size = max(12, min(28, 0.28 * len(matrix.columns)))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(matrix.to_numpy(dtype=float), vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title("Pairwise cross-marginal overlap")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(matrix.index, fontsize=6)
    fig.colorbar(image, ax=ax, label="overlap")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_cross_marginal_review(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    common_columns: list[str],
    output_dir: Path,
    numeric_bins: int,
    plot_count: int,
    top_values: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    real_bucketed, synthetic_bucketed = bucketed_columns(
        real,
        synthetic,
        common_columns,
        numeric_bins,
    )

    rows = []
    overlap_matrix = pd.DataFrame(
        np.eye(len(common_columns)),
        index=common_columns,
        columns=common_columns,
    )
    for column_a, column_b in tqdm(
        list(combinations(common_columns, 2)),
        desc="validating cross-marginals",
        unit="pair",
    ):
        real_counts = joint_counts(real_bucketed[column_a], real_bucketed[column_b])
        synthetic_counts = joint_counts(
            synthetic_bucketed[column_a],
            synthetic_bucketed[column_b],
        )
        overlap = distribution_overlap(real_counts, synthetic_counts)
        overlap_matrix.loc[column_a, column_b] = overlap
        overlap_matrix.loc[column_b, column_a] = overlap
        rows.append(
            {
                "column_a": column_a,
                "column_b": column_b,
                "kind_a": "numeric" if column_a in NUMERIC_COLUMNS else "categorical",
                "kind_b": "numeric" if column_b in NUMERIC_COLUMNS else "categorical",
                "real_unique_pairs": len(real_counts),
                "synthetic_unique_pairs": len(synthetic_counts),
                "cross_marginal_overlap": f"{overlap:.8f}",
                "total_variation_distance": f"{1 - overlap:.8f}",
            }
        )

    rows.sort(key=lambda row: float(row["cross_marginal_overlap"]))
    summary_path = output_dir / "cross_marginal_overlaps.csv"
    with summary_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "column_a",
                "column_b",
                "kind_a",
                "kind_b",
                "real_unique_pairs",
                "synthetic_unique_pairs",
                "cross_marginal_overlap",
                "total_variation_distance",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    overlap_matrix.to_csv(output_dir / "cross_marginal_overlap_matrix.csv")
    plot_cross_marginal_heatmap(
        overlap_matrix,
        output_dir / "cross_marginal_overlap_heatmap.png",
    )
    clear_cross_marginal_pair_plots(output_dir)

    for row in tqdm(
        rows[:plot_count],
        desc="charting worst cross-marginals",
        unit="chart",
    ):
        column_a = row["column_a"]
        column_b = row["column_b"]
        real_counts = joint_counts(real_bucketed[column_a], real_bucketed[column_b])
        synthetic_counts = joint_counts(
            synthetic_bucketed[column_a],
            synthetic_bucketed[column_b],
        )
        plot_cross_marginal(
            column_a,
            column_b,
            real_counts,
            synthetic_counts,
            output_dir,
            top_values,
        )

    print(f"wrote cross-marginal review to {output_dir}")


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
                column,
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

    write_cross_marginal_review(
        real,
        synthetic,
        common_columns,
        args.cross_marginal_dir,
        args.numeric_bins,
        args.cross_marginal_plots,
        args.cross_marginal_top_values,
    )

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
