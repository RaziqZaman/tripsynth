#!/usr/bin/env python3
"""Count values in each distilled survey column."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


DEFAULT_INPUT_CSV = Path("01x_distilled-survey.csv")
DEFAULT_OUTPUT_CSV = Path("01y_distilled-stats.csv")
BLANK_VALUE = "<blank>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", nargs="?", default=DEFAULT_INPUT_CSV)
    parser.add_argument("-o", "--output-csv", default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    counters: dict[str, Counter[str]] = {}
    row_count = 0

    with input_csv.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise SystemExit(f"{input_csv} has no header")

        counters = {column: Counter() for column in reader.fieldnames}
        for row in reader:
            row_count += 1
            for column in reader.fieldnames:
                value = row.get(column, "")
                if value == "":
                    value = BLANK_VALUE
                counters[column][value] += 1

    with output_csv.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["column", "value", "count", "percentage"],
        )
        writer.writeheader()
        for column, counter in counters.items():
            for value, count in counter.most_common():
                percentage = count / row_count if row_count else 0
                writer.writerow(
                    {
                        "column": column,
                        "value": value,
                        "count": count,
                        "percentage": f"{percentage:.6%}",
                    }
                )

    print(f"wrote {output_csv}")
    print(f"rows: {row_count}")
    print(f"columns: {len(counters)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
