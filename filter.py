#!/usr/bin/env python3
"""Filter synthetic_trips parquet data to a fixed set of columns."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


KEEP_COLUMNS = [
    "household_id",
    "person_id",
    "tripid",
    "tripno",
    "o_activity",
    "d_activity",
    "fueltype",
    "departure_time_hhmm",
    "arrival_time_hhmm",
    "travel_mode",
    "home_state_county_fips",
    "reported_travel_time",
    "hh_income_detailed",
    "home_type",
    "numworkers",
    "numbicycle",
    "j1_telecommute",
    "j1_benefits_ev_charging",
    "home_tpb_taz",
    "home_bmc_taz",
    "home_tract_fips",
    "o_state_county_fips",
    "d_state_county_fips",
    "o_tract_fips",
    "d_tract_fips",
    "o_tpb_taz",
    "d_tpb_taz",
    "o_bmc_taz",
    "d_bmc_taz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter synthetic_trips parquet file to a fixed column subset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("synthetic_trips.parquet"),
        help="Input parquet file (default: synthetic_trips.parquet).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("synthetic_trips_filtered.parquet"),
        help="Output parquet file (default: synthetic_trips_filtered.parquet).",
    )
    return parser.parse_args()


def filter_parquet(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    table = pq.read_table(input_path)
    input_columns = set(table.column_names)
    missing = [col for col in KEEP_COLUMNS if col not in input_columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Input parquet is missing required column(s): {missing_str}")

    filtered = table.select(KEEP_COLUMNS)
    pq.write_table(filtered, output_path, compression="zstd")

    print(
        f"Wrote {output_path} with {filtered.num_rows} rows and "
        f"{filtered.num_columns} columns."
    )


def main() -> int:
    args = parse_args()
    filter_parquet(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
