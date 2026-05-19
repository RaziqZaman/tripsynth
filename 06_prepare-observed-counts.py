#!/usr/bin/env python3
"""Normalize raw observed traffic-count records for Stage 1 validation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


RAW_COUNTS = Path("06x_stage1/mdot_aadt_points_raw.csv")
OUTPUT_COUNTS = Path("06x_stage1/observed_counts_points.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-counts", type=Path, default=RAW_COUNTS)
    parser.add_argument("--out", type=Path, default=OUTPUT_COUNTS)
    parser.add_argument("--count-id-column", default="LOCATION_ID")
    parser.add_argument("--fallback-id-column", default="OBJECTID")
    parser.add_argument("--observed-column", default="AADT")
    parser.add_argument("--x-column", default="geometry_x")
    parser.add_argument("--y-column", default="geometry_y")
    return parser.parse_args()


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def first_present(row: dict[str, str], names: list[str]) -> str:
    for name in names:
        value = row.get(name, "")
        if value:
            return value
    return ""


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    output_rows = []
    with args.raw_counts.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            observed_volume = parse_float(row.get(args.observed_column, ""))
            lon = parse_float(row.get(args.x_column, ""))
            lat = parse_float(row.get(args.y_column, ""))
            if observed_volume <= 0 or lon == 0 or lat == 0:
                continue

            count_id = first_present(
                row,
                [args.count_id_column, args.fallback_id_column],
            )
            output_rows.append(
                {
                    "count_id": count_id,
                    "observed_volume": f"{observed_volume:.8f}",
                    "lon": f"{lon:.8f}",
                    "lat": f"{lat:.8f}",
                    "source_observed_column": args.observed_column,
                    "route_id": row.get("ROUTEID", ""),
                    "road_name": row.get("ROADNAME", ""),
                    "county": row.get("COUNTY_DESC", ""),
                    "station_desc": row.get("STATION_DESC", ""),
                    "location_id": row.get("LOCATION_ID", ""),
                    "object_id": row.get("OBJECTID", ""),
                }
            )

    with args.out.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "count_id",
                "observed_volume",
                "lon",
                "lat",
                "source_observed_column",
                "route_id",
                "road_name",
                "county",
                "station_desc",
                "location_id",
                "object_id",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"wrote {args.out}")
    print(f"observed count points: {len(output_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
