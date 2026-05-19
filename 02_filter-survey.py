#!/usr/bin/env python3
"""Filter the distilled survey CSV for modeling-ready rows."""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path


INPUT_CSV = Path("01x_distilled-survey.csv")
OUTPUT_CSV = Path("02x_filtered-survey.csv")

BASE_DATE = date(2017, 1, 1)
LOW_BLANK_THRESHOLD = 0.04

REMOVE_COLUMNS = {
    "arrival_time_hhmm",
    "td_paidpark",
    "td_hov",
    "td_toll",
}

REQUIRED_TPB_TAZ_COLUMNS = {
    "o_tpb_taz",
    "d_tpb_taz",
    "home_tpb_taz",
}


def is_removed_column(column: str) -> bool:
    if column in REMOVE_COLUMNS:
        return True
    if column.endswith("tract_fips"):
        return True
    if column.endswith("bmc_taz"):
        return True
    return False


def output_column_name(column: str) -> str:
    if column == "departure_time_hhmm":
        return "departure_time_min"
    if column == "tdate_string":
        return "tdate_days"
    return column


def hhmm_to_minutes(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    hour_text, minute_text = value.split(":", 1)
    return str(int(hour_text) * 60 + int(minute_text))


def date_to_days(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    parsed = datetime.strptime(value, "%Y-%m-%d").date()
    return str((parsed - BASE_DATE).days)


def transform_value(column: str, value: str) -> str:
    if column == "departure_time_hhmm":
        return hhmm_to_minutes(value)
    if column == "tdate_string":
        return date_to_days(value)
    return value


def has_blank(row: dict[str, str], columns: set[str]) -> bool:
    return any(row.get(column, "").strip() == "" for column in columns)


def main() -> int:
    with INPUT_CSV.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise SystemExit(f"{INPUT_CSV} has no header")

        input_columns = [column for column in reader.fieldnames if not is_removed_column(column)]
        output_columns = [output_column_name(column) for column in input_columns]
        rows = list(reader)

    total_rows = len(rows)
    low_blank_columns = set()
    for column in input_columns:
        blank_count = sum(1 for row in rows if row.get(column, "").strip() == "")
        if total_rows and blank_count / total_rows < LOW_BLANK_THRESHOLD:
            low_blank_columns.add(column)

    required_columns = low_blank_columns | REQUIRED_TPB_TAZ_COLUMNS
    missing_required = sorted(required_columns - set(input_columns))
    if missing_required:
        raise SystemExit(f"Missing required columns after filtering: {missing_required}")

    kept_rows = []
    for row in rows:
        if has_blank(row, required_columns):
            continue
        kept_rows.append(
            {
                output_column_name(column): transform_value(column, row.get(column, ""))
                for column in input_columns
            }
        )

    with OUTPUT_CSV.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(kept_rows)

    print(f"wrote {OUTPUT_CSV}")
    print(f"input rows: {total_rows}")
    print(f"output rows: {len(kept_rows)}")
    print(f"columns: {len(output_columns)}")
    print(f"low-blank required columns: {len(low_blank_columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
