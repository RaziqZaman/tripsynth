#!/usr/bin/env python3
"""Consolidate the flat survey CSV into the WCTR upload shape."""

from __future__ import annotations

import csv
from pathlib import Path


INPUT_CSV = Path("combined-flat-survey.csv")
FALLBACK_INPUT_CSV = Path("00__combined-flat-survey.csv")
OUTPUT_CSV = Path("00x_consolidated-survey.csv")

REMOVE_COLUMNS = {
    "travelers_total",
    "mode_hh_vehicle",
    "wttrdfin",
    "wtperfin",
    "vehnum",
    "in_tpb",
    "in_bmc",
    "person_id",
    "tripid",
    "hh_income_broad",
    "age_group",
    "trips_yesno",
    "no_travel",
}


def clean_digits(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if "." in value:
        value = value.split(".", 1)[0]
    return "".join(char for char in value if char.isdigit())


def padded_digits(value: str, width: int) -> str:
    digits = clean_digits(value)
    if not digits:
        return ""
    return digits.zfill(width)[-width:]


def combined_tract_fips(row: dict[str, str], county_column: str, tract_column: str) -> str:
    county = padded_digits(row.get(county_column, ""), 5)
    tract = padded_digits(row.get(tract_column, ""), 6)
    if not county or not tract:
        return ""
    return f"{county}{tract}"


def hhmm_to_time(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if ":" in value:
        hour, minute, *_ = value.split(":") + [""]
        return f"{int(hour):02d}:{int(minute):02d}"

    digits = clean_digits(value)
    if not digits:
        return value
    digits = digits.zfill(4)
    return f"{int(digits[:-2]):02d}:{int(digits[-2:]):02d}"


def cap_td_telecommute_time(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if value == "8+":
        return "8"
    try:
        number = float(value.rstrip("+"))
    except ValueError:
        return value
    if number >= 8:
        return "8"
    return value


def build_output_columns(input_columns: list[str]) -> tuple[list[str], dict[str, str]]:
    tract_to_county = {}
    for column in input_columns:
        if not column.endswith("_tract_fips"):
            continue
        prefix = column[: -len("_tract_fips")]
        county_column = f"{prefix}_state_county_fips"
        if county_column in input_columns:
            tract_to_county[column] = county_column

    output_columns = []
    county_columns = set(tract_to_county.values())
    for column in input_columns:
        if column.endswith("_state_fips"):
            continue
        if column in REMOVE_COLUMNS:
            continue
        if column in county_columns:
            continue
        output_columns.append(column)

    return output_columns, tract_to_county


def transform_row(
    row: dict[str, str],
    output_columns: list[str],
    tract_to_county: dict[str, str],
) -> dict[str, str]:
    output = {}
    for column in output_columns:
        value = row.get(column, "")
        if column in tract_to_county:
            value = combined_tract_fips(row, tract_to_county[column], column)
        elif column.endswith("_tract_fips"):
            value = padded_digits(value, 6)
        elif column.startswith(("departure_time", "arrival_time")):
            value = hhmm_to_time(value)
        elif column == "td_telecommute_time":
            value = cap_td_telecommute_time(value)
        output[column] = value
    return output


def main() -> int:
    input_csv = INPUT_CSV if INPUT_CSV.exists() else FALLBACK_INPUT_CSV
    if not input_csv.exists():
        raise FileNotFoundError(
            f"missing input CSV; expected {INPUT_CSV} or {FALLBACK_INPUT_CSV}"
        )

    with input_csv.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise SystemExit(f"{INPUT_CSV} has no header")

        output_columns, tract_to_county = build_output_columns(reader.fieldnames)

        with OUTPUT_CSV.open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=output_columns)
            writer.writeheader()
            for row in reader:
                writer.writerow(transform_row(row, output_columns, tract_to_county))

    print(f"read {input_csv}")
    print(f"wrote {OUTPUT_CSV}")
    print(f"columns: {len(output_columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
