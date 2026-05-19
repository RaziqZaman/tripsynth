#!/usr/bin/env python3
"""Fill remaining blanks in the filtered survey CSV."""

from __future__ import annotations

import csv
from pathlib import Path


INPUT_CSV = Path("02x_filtered-survey.csv")
OUTPUT_CSV = Path("03x_filled-survey.csv")

COLUMN_FILL_VALUES = {
    "smartphone": "2",
    "employment_status": "6",
    "j1_commute_mode": "9",
    "volunteer_status": "1",
}

ZERO_FILL_COLUMNS = {
    "vehicle_occupancy",
    "park_pay",
    "subway_used",
    "hov_used",
    "toll_road_used",
    "jobs_count",
    "j1_telecommute_days",
    "td_telecommute_time",
    "walk_bike_loop_trips",
}


def fill_value(column: str, value: str) -> str:
    value = value.strip()
    if value == "":
        if column in COLUMN_FILL_VALUES:
            value = COLUMN_FILL_VALUES[column]
        elif column in ZERO_FILL_COLUMNS:
            value = "0"
        else:
            value = "N/A"

    if column == "j1_telecommute_days":
        try:
            if float(value.rstrip("+")) >= 5:
                return "5"
        except ValueError:
            return value

    return value


def main() -> int:
    with INPUT_CSV.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise SystemExit(f"{INPUT_CSV} has no header")

        fieldnames = reader.fieldnames
        with OUTPUT_CSV.open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            row_count = 0
            for row in reader:
                writer.writerow(
                    {column: fill_value(column, row.get(column, "")) for column in fieldnames}
                )
                row_count += 1

    print(f"wrote {OUTPUT_CSV}")
    print(f"rows: {row_count}")
    print(f"columns: {len(fieldnames)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
