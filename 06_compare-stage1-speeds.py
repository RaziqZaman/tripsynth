#!/usr/bin/env python3
"""Compare MATSim link speeds against observed VDOT 511 segment speeds."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DEFAULT_DIR = Path("06x_stage1_md_internal")
DEFAULT_NETWORK = Path("06x_stage1_osm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed-speeds", type=Path, default=DEFAULT_NETWORK / "observed_speeds.csv")
    parser.add_argument("--real-link-speeds", type=Path, required=True)
    parser.add_argument("--synthetic-link-speeds", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_DIR / "speed_validation.csv")
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_DIR / "speed_validation_summary.csv")
    return parser.parse_args()


def parse_float(value: str | None, default: float = math.nan) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_link_speeds(path: Path) -> dict[str, dict[str, float]]:
    speeds: dict[str, dict[str, float]] = {}
    with path.open(newline="") as input_file:
        for row in csv.DictReader(input_file):
            link_id = row.get("link_id", "")
            if not link_id:
                continue
            speeds[link_id] = {
                "mean_speed_mph": parse_float(row.get("mean_speed_mph")),
                "traversals": parse_float(row.get("traversals"), 0.0),
            }
    return speeds


def correlation(pairs: list[tuple[float, float]]) -> float:
    if len(pairs) < 2:
        return math.nan
    xs = [model for model, _ in pairs]
    ys = [observed for _, observed in pairs]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_denom = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_denom = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_denom == 0 or y_denom == 0:
        return math.nan
    return numerator / (x_denom * y_denom)


def scenario_rows(
    scenario: str,
    observed_rows: list[dict[str, str]],
    link_speeds: dict[str, dict[str, float]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in observed_rows:
        observed = parse_float(row.get("observed_speed_mph"))
        link_id = row.get("matsim_link_id", "")
        model = link_speeds.get(link_id, {})
        model_speed = model.get("mean_speed_mph", math.nan)
        traversals = model.get("traversals", 0.0)
        error = model_speed - observed if not math.isnan(model_speed) and not math.isnan(observed) else math.nan
        percent_error = error / observed if observed and not math.isnan(error) else math.nan
        rows.append(
            {
                "scenario": scenario,
                "segment_id": row.get("segment_id", ""),
                "matsim_link_id": link_id,
                "observed_speed_mph": f"{observed:.8f}" if not math.isnan(observed) else "",
                "model_speed_mph": f"{model_speed:.8f}" if not math.isnan(model_speed) else "",
                "error_mph": f"{error:.8f}" if not math.isnan(error) else "",
                "absolute_error_mph": f"{abs(error):.8f}" if not math.isnan(error) else "",
                "percent_error": f"{percent_error:.8f}" if not math.isnan(percent_error) else "",
                "model_traversals": f"{traversals:.0f}",
                "match_distance_m": row.get("match_distance_m", ""),
                "route": row.get("route", ""),
                "description": row.get("description", ""),
                "observed_direction": row.get("observed_direction", ""),
                "matched_direction": row.get("matched_direction", ""),
                "congestion": row.get("congestion", ""),
                "imputation": row.get("imputation", ""),
                "fetched_at_utc": row.get("fetched_at_utc", ""),
            }
        )
    return rows


def summary_row(scenario: str, rows: list[dict[str, str]]) -> dict[str, str]:
    modeled = []
    for row in rows:
        observed = parse_float(row.get("observed_speed_mph"))
        model = parse_float(row.get("model_speed_mph"))
        if not math.isnan(observed) and not math.isnan(model):
            modeled.append((model, observed))
    errors = [model - observed for model, observed in modeled]
    abs_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]
    nonzero = [(model, observed) for model, observed in modeled if observed]
    return {
        "scenario": scenario,
        "matched_segments": str(len(rows)),
        "modeled_segments": str(len(modeled)),
        "missing_model_segments": str(len(rows) - len(modeled)),
        "observed_mean_speed_mph": f"{(sum(observed for _, observed in modeled) / len(modeled)) if modeled else math.nan:.8f}",
        "model_mean_speed_mph": f"{(sum(model for model, _ in modeled) / len(modeled)) if modeled else math.nan:.8f}",
        "mean_bias_mph": f"{(sum(errors) / len(errors)) if errors else math.nan:.8f}",
        "mae_mph": f"{(sum(abs_errors) / len(abs_errors)) if abs_errors else math.nan:.8f}",
        "rmse_mph": f"{math.sqrt(sum(squared_errors) / len(squared_errors)) if squared_errors else math.nan:.8f}",
        "mape": f"{(sum(abs(model - observed) / observed for model, observed in nonzero) / len(nonzero)) if nonzero else math.nan:.8f}",
        "correlation": f"{correlation(modeled):.8f}",
        "within_5_mph_share": f"{(sum(abs(error) <= 5 for error in errors) / len(errors)) if errors else math.nan:.8f}",
        "within_10_mph_share": f"{(sum(abs(error) <= 10 for error in errors) / len(errors)) if errors else math.nan:.8f}",
    }


def main() -> int:
    args = parse_args()
    with args.observed_speeds.open(newline="") as input_file:
        observed_rows = list(csv.DictReader(input_file))
    real_speeds = read_link_speeds(args.real_link_speeds)
    synthetic_speeds = read_link_speeds(args.synthetic_link_speeds)

    rows = scenario_rows("real", observed_rows, real_speeds) + scenario_rows("synthetic", observed_rows, synthetic_speeds)
    fieldnames = [
        "scenario", "segment_id", "matsim_link_id", "observed_speed_mph", "model_speed_mph",
        "error_mph", "absolute_error_mph", "percent_error", "model_traversals", "match_distance_m",
        "route", "description", "observed_direction", "matched_direction", "congestion", "imputation", "fetched_at_utc",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summaries = [
        summary_row("real", [row for row in rows if row["scenario"] == "real"]),
        summary_row("synthetic", [row for row in rows if row["scenario"] == "synthetic"]),
    ]
    with args.summary_out.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    print(f"observed segments: {len(observed_rows):,}")
    print(f"wrote {args.out}")
    print(f"wrote {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
