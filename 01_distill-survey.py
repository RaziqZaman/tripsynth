#!/usr/bin/env python3
"""Distill the consolidated survey CSV to the SRUVEY upload shape."""

from __future__ import annotations

import csv
from pathlib import Path


INPUT_CSV = Path("00x_consolidated-survey.csv")
OUTPUT_CSV = Path("01x_distilled_survey.csv")

REMOVE_COLUMNS = {
    "household_id",
    "persno",
    "tripno",
    "home_state_puma",
    "area_type",
}


def keep_column(column: str) -> bool:
    if column in REMOVE_COLUMNS:
        return False
    if column.startswith("hh_member"):
        return False
    if column.endswith("_imp"):
        return False
    return True


def main() -> int:
    with INPUT_CSV.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise SystemExit(f"{INPUT_CSV} has no header")

        output_columns = [column for column in reader.fieldnames if keep_column(column)]

        with OUTPUT_CSV.open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=output_columns)
            writer.writeheader()
            for row in reader:
                writer.writerow({column: row[column] for column in output_columns})

    print(f"wrote {OUTPUT_CSV}")
    print(f"columns: {len(output_columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
