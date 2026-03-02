#!/usr/bin/env python3
"""Create per-column value count CSVs from tupled-survey.parquet."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from tabular_io import read_rows

INPUT_TABLE = Path("tupled-survey.parquet")
OUTPUT_DIR = Path("tupled-counts")
NULL_LABEL = "<NULL>"


def normalize_value(raw: object) -> str:
    if raw is None:
        return NULL_LABEL
    value = str(raw).strip()
    return NULL_LABEL if value == "" else value


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned or "unnamed_column"


def main() -> int:
    if not INPUT_TABLE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_TABLE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    columns, rows = read_rows(INPUT_TABLE)
    if not columns:
        raise ValueError("Input table has no header row.")
    counters: dict[str, Counter[str]] = {col: Counter() for col in columns}
    for row in rows:
        for col in columns:
            counters[col][normalize_value(row.get(col))] += 1

    used_names: dict[str, int] = {}
    for col in columns:
        base = safe_filename(col)
        idx = used_names.get(base, 0)
        used_names[base] = idx + 1
        filename = f"{base}.csv" if idx == 0 else f"{base}_{idx}.csv"
        out_path = OUTPUT_DIR / filename

        with out_path.open("w", newline="", encoding="utf-8") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(["value", "count"])
            for value, count in counters[col].most_common():
                writer.writerow([value, count])

    print(f"Wrote {len(columns)} files to {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
