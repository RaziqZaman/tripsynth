#!/usr/bin/env python3
"""Prepare Stage 1 vehicle-trip demand for assignment/simulation comparison."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REAL_CSV = Path("03x_filled-survey.csv")
SYNTHETIC_CSV = Path("04x_synthetic-trips.csv")
OUTPUT_DIR = Path("06x_stage1")
DEFAULT_SYNTHETIC_SCALE = 12.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-csv", type=Path, default=REAL_CSV)
    parser.add_argument("--synthetic-csv", type=Path, default=SYNTHETIC_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--synthetic-scale", type=float, default=DEFAULT_SYNTHETIC_SCALE)
    parser.add_argument(
        "--road-modes",
        default="",
        help=(
            "optional comma-separated travel_mode values to include; "
            "blank keeps every row with positive vehicle_occupancy"
        ),
    )
    return parser.parse_args()


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_road_modes(value: str) -> set[str] | None:
    modes = {item.strip() for item in value.split(",") if item.strip()}
    return modes or None


def include_row(row: dict[str, str], road_modes: set[str] | None) -> bool:
    occupancy = parse_float(row.get("vehicle_occupancy", "0"))
    if occupancy <= 0:
        return False
    if road_modes is None:
        return True
    return row.get("travel_mode", "") in road_modes


def iter_vehicle_rows(
    path: Path,
    scenario: str,
    person_weight_column: str | None,
    default_person_weight: float,
    road_modes: set[str] | None,
) -> tuple[list[dict[str, str]], dict[str, float]]:
    rows: list[dict[str, str]] = []
    totals = {
        "input_rows": 0.0,
        "included_rows": 0.0,
        "person_trips": 0.0,
        "unscaled_vehicle_trips": 0.0,
    }

    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        for row_id, row in enumerate(reader, start=1):
            totals["input_rows"] += 1
            if not include_row(row, road_modes):
                continue

            occupancy = parse_float(row.get("vehicle_occupancy", "0"))
            person_weight = (
                parse_float(row.get(person_weight_column, ""), 0.0)
                if person_weight_column
                else default_person_weight
            )
            vehicle_weight = person_weight / occupancy
            totals["included_rows"] += 1
            totals["person_trips"] += person_weight
            totals["unscaled_vehicle_trips"] += vehicle_weight

            rows.append(
                {
                    "scenario": scenario,
                    "source_row": str(row_id),
                    "o_tpb_taz": row.get("o_tpb_taz", ""),
                    "d_tpb_taz": row.get("d_tpb_taz", ""),
                    "o_activity": row.get("o_activity", ""),
                    "d_activity": row.get("d_activity", ""),
                    "departure_time_min": row.get("departure_time_min", ""),
                    "travel_mode": row.get("travel_mode", ""),
                    "vehicle_occupancy": row.get("vehicle_occupancy", ""),
                    "person_weight": f"{person_weight:.8f}",
                    "vehicle_weight_unscaled": f"{vehicle_weight:.8f}",
                }
            )

    return rows, totals


def write_vehicle_trips(path: Path, rows: list[dict[str, str]], scale: float) -> None:
    fieldnames = [
        "scenario",
        "source_row",
        "o_tpb_taz",
        "d_tpb_taz",
        "o_activity",
        "d_activity",
        "departure_time_min",
        "travel_mode",
        "vehicle_occupancy",
        "person_weight",
        "vehicle_weight_unscaled",
        "vehicle_weight",
    ]
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            vehicle_weight = parse_float(row["vehicle_weight_unscaled"]) * scale
            writer.writerow({**row, "vehicle_weight": f"{vehicle_weight:.8f}"})


def write_totals(
    path: Path,
    real_totals: dict[str, float],
    synthetic_totals: dict[str, float],
    real_scale: float,
    synthetic_scale: float,
    road_modes: set[str] | None,
) -> None:
    rows = [
        {
            "scenario": "real",
            "input_rows": real_totals["input_rows"],
            "included_rows": real_totals["included_rows"],
            "person_trips_unscaled": real_totals["person_trips"],
            "vehicle_trips_unscaled": real_totals["unscaled_vehicle_trips"],
            "scale": real_scale,
            "vehicle_trips_scaled": real_totals["unscaled_vehicle_trips"] * real_scale,
        },
        {
            "scenario": "synthetic",
            "input_rows": synthetic_totals["input_rows"],
            "included_rows": synthetic_totals["included_rows"],
            "person_trips_unscaled": synthetic_totals["person_trips"],
            "vehicle_trips_unscaled": synthetic_totals["unscaled_vehicle_trips"],
            "scale": synthetic_scale,
            "vehicle_trips_scaled": synthetic_totals["unscaled_vehicle_trips"]
            * synthetic_scale,
        },
    ]

    with path.open("w", newline="") as output_file:
        fieldnames = [
            "scenario",
            "input_rows",
            "included_rows",
            "person_trips_unscaled",
            "vehicle_trips_unscaled",
            "scale",
            "vehicle_trips_scaled",
            "road_modes",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        road_modes_text = ",".join(sorted(road_modes)) if road_modes else "positive_occupancy"
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "input_rows": f"{row['input_rows']:.0f}",
                    "included_rows": f"{row['included_rows']:.0f}",
                    "person_trips_unscaled": f"{row['person_trips_unscaled']:.8f}",
                    "vehicle_trips_unscaled": f"{row['vehicle_trips_unscaled']:.8f}",
                    "scale": f"{row['scale']:.12f}",
                    "vehicle_trips_scaled": f"{row['vehicle_trips_scaled']:.8f}",
                    "road_modes": road_modes_text,
                }
            )


def main() -> int:
    args = parse_args()
    road_modes = parse_road_modes(args.road_modes)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    real_rows, real_totals = iter_vehicle_rows(
        args.real_csv,
        "real",
        person_weight_column="wthhfin",
        default_person_weight=1.0,
        road_modes=road_modes,
    )
    synthetic_rows, synthetic_totals = iter_vehicle_rows(
        args.synthetic_csv,
        "synthetic",
        person_weight_column=None,
        default_person_weight=1.0,
        road_modes=road_modes,
    )

    synthetic_vehicle_trips_scaled = (
        synthetic_totals["unscaled_vehicle_trips"] * args.synthetic_scale
    )
    real_vehicle_trips = real_totals["unscaled_vehicle_trips"]
    real_scale = (
        synthetic_vehicle_trips_scaled / real_vehicle_trips
        if real_vehicle_trips
        else 0.0
    )

    write_vehicle_trips(args.output_dir / "real_vehicle_trips.csv", real_rows, real_scale)
    write_vehicle_trips(
        args.output_dir / "synthetic_vehicle_trips.csv",
        synthetic_rows,
        args.synthetic_scale,
    )
    write_totals(
        args.output_dir / "scenario_totals.csv",
        real_totals,
        synthetic_totals,
        real_scale,
        args.synthetic_scale,
        road_modes,
    )

    print(f"wrote {args.output_dir / 'real_vehicle_trips.csv'}")
    print(f"wrote {args.output_dir / 'synthetic_vehicle_trips.csv'}")
    print(f"wrote {args.output_dir / 'scenario_totals.csv'}")
    print(
        "scaled vehicle trips: "
        f"real={real_totals['unscaled_vehicle_trips'] * real_scale:.2f}, "
        f"synthetic={synthetic_vehicle_trips_scaled:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
