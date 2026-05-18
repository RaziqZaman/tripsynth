#!/usr/bin/env python3
"""Print a compact summary of the combined flat survey CSV."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


DEFAULT_COLUMNS = (
    "travel_mode",
    "o_activity",
    "d_activity",
    "hhsize",
    "age_group",
    "gender",
    "employment_status",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", default="combined-flat-survey.csv")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--columns", nargs="*", default=list(DEFAULT_COLUMNS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path)

    if not csv_path.exists():
        print(f"Missing CSV: {csv_path}")
        return 1

    counters = {column: Counter() for column in args.columns}
    row_count = 0

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        available = [column for column in args.columns if column in fieldnames]
        missing = [column for column in args.columns if column not in fieldnames]
        counters = {column: counters[column] for column in available}

        for row in reader:
            row_count += 1
            for column in available:
                value = row.get(column, "").strip() or "<blank>"
                counters[column][value] += 1

    print(f"file: {csv_path}")
    print(f"rows: {row_count:,}")
    print(f"columns: {len(fieldnames):,}")

    if missing:
        print(f"missing requested columns: {', '.join(missing)}")

    for column, counter in counters.items():
        print()
        print(column)
        for value, count in counter.most_common(args.top):
            share = count / row_count if row_count else 0
            print(f"  {value}: {count:,} ({share:.1%})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
