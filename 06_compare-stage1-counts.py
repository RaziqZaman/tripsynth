#!/usr/bin/env python3
"""Compare Stage 1 assigned link volumes against observed traffic counts."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


OUTPUT_DIR = Path("06x_stage1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed-counts", type=Path, required=True)
    parser.add_argument("--real-assigned", type=Path, required=True)
    parser.add_argument("--synthetic-assigned", type=Path, required=True)
    parser.add_argument("--scenario-totals", type=Path, default=OUTPUT_DIR / "scenario_totals.csv")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "count_validation.csv")
    parser.add_argument("--summary-out", type=Path, default=OUTPUT_DIR / "count_validation_summary.csv")
    parser.add_argument("--key-column", default="link_id")
    parser.add_argument("--observed-column", default="observed_volume")
    parser.add_argument("--assigned-column", default="volume")
    parser.add_argument(
        "--observed-scale",
        type=float,
        default=None,
        help="scale observed counts to the simulated demand fraction; defaults to real scale in scenario_totals",
    )
    return parser.parse_args()


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_observed_scale(path: Path) -> float | None:
    if not path.exists():
        return None
    with path.open(newline="") as input_file:
        for row in csv.DictReader(input_file):
            if row.get("scenario") == "real":
                return parse_float(row.get("scale", ""), default=0.0)
    return None


def read_values(path: Path, key_column: str, value_column: str) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            key = row.get(key_column, "")
            if not key:
                continue
            values[key] = values.get(key, 0.0) + parse_float(row.get(value_column, ""))
    return values


def geh(model: float, observed: float) -> float:
    if model + observed <= 0:
        return 0.0
    return math.sqrt(2.0 * (model - observed) ** 2 / (model + observed))


def correlation(pairs: list[tuple[float, float]]) -> float:
    if len(pairs) < 2:
        return float("nan")
    model_values = [model for model, _ in pairs]
    observed_values = [observed for _, observed in pairs]
    model_mean = sum(model_values) / len(model_values)
    observed_mean = sum(observed_values) / len(observed_values)
    numerator = sum(
        (model - model_mean) * (observed - observed_mean)
        for model, observed in pairs
    )
    model_denom = math.sqrt(sum((model - model_mean) ** 2 for model in model_values))
    observed_denom = math.sqrt(
        sum((observed - observed_mean) ** 2 for observed in observed_values)
    )
    if model_denom == 0 or observed_denom == 0:
        return float("nan")
    return numerator / (model_denom * observed_denom)


def scenario_rows(
    scenario: str,
    observed: dict[str, float],
    assigned: dict[str, float],
) -> list[dict[str, str]]:
    rows = []
    for key in sorted(set(observed) & set(assigned)):
        observed_value = observed[key]
        model_value = assigned[key]
        error = model_value - observed_value
        absolute_error = abs(error)
        percent_error = error / observed_value if observed_value else float("nan")
        rows.append(
            {
                "scenario": scenario,
                "link_id": key,
                "observed_volume": f"{observed_value:.8f}",
                "model_volume": f"{model_value:.8f}",
                "error": f"{error:.8f}",
                "absolute_error": f"{absolute_error:.8f}",
                "percent_error": f"{percent_error:.8f}",
                "geh": f"{geh(model_value, observed_value):.8f}",
            }
        )
    return rows


def summary_row(scenario: str, rows: list[dict[str, str]]) -> dict[str, str]:
    pairs = [
        (parse_float(row["model_volume"]), parse_float(row["observed_volume"]))
        for row in rows
    ]
    errors = [model - observed for model, observed in pairs]
    abs_errors = [abs(error) for error in errors]
    squared_errors = [error**2 for error in errors]
    nonzero_observed = [
        (model, observed) for model, observed in pairs if observed != 0
    ]
    mape = (
        sum(abs(model - observed) / observed for model, observed in nonzero_observed)
        / len(nonzero_observed)
        if nonzero_observed
        else float("nan")
    )
    geh_values = [parse_float(row["geh"]) for row in rows]
    return {
        "scenario": scenario,
        "matched_counts": str(len(rows)),
        "observed_total": f"{sum(observed for _, observed in pairs):.8f}",
        "model_total": f"{sum(model for model, _ in pairs):.8f}",
        "mean_bias": f"{(sum(errors) / len(errors)) if errors else float('nan'):.8f}",
        "mae": f"{(sum(abs_errors) / len(abs_errors)) if abs_errors else float('nan'):.8f}",
        "rmse": f"{math.sqrt(sum(squared_errors) / len(squared_errors)) if squared_errors else float('nan'):.8f}",
        "mape": f"{mape:.8f}",
        "correlation": f"{correlation(pairs):.8f}",
        "geh_lt_5_share": f"{(sum(value < 5 for value in geh_values) / len(geh_values)) if geh_values else float('nan'):.8f}",
    }


def main() -> int:
    args = parse_args()
    observed_scale = args.observed_scale
    if observed_scale is None:
        observed_scale = read_observed_scale(args.scenario_totals)
    if observed_scale is None:
        observed_scale = 1.0

    observed_raw = read_values(
        args.observed_counts,
        args.key_column,
        args.observed_column,
    )
    observed = {
        key: value * observed_scale
        for key, value in observed_raw.items()
    }
    real_assigned = read_values(
        args.real_assigned,
        args.key_column,
        args.assigned_column,
    )
    synthetic_assigned = read_values(
        args.synthetic_assigned,
        args.key_column,
        args.assigned_column,
    )

    rows = scenario_rows("real", observed, real_assigned) + scenario_rows(
        "synthetic",
        observed,
        synthetic_assigned,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as output_file:
        fieldnames = [
            "scenario",
            "link_id",
            "observed_volume",
            "model_volume",
            "error",
            "absolute_error",
            "percent_error",
            "geh",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    real_rows = [row for row in rows if row["scenario"] == "real"]
    synthetic_rows = [row for row in rows if row["scenario"] == "synthetic"]
    summaries = [
        summary_row("real", real_rows),
        summary_row("synthetic", synthetic_rows),
    ]
    with args.summary_out.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    print(f"observed scale: {observed_scale:.12f}")
    print(f"wrote {args.out}")
    print(f"wrote {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
