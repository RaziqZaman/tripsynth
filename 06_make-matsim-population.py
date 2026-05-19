#!/usr/bin/env python3
"""Convert Stage 1 weighted vehicle trips into a simple MATSim population XML."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import random
from pathlib import Path
from typing import TextIO
from xml.sax.saxutils import escape


OUTPUT_DIR = Path("06x_stage1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vehicle-trips", type=Path, required=True)
    parser.add_argument("--taz-centroids", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--taz-column", default="taz")
    parser.add_argument("--x-column", default="x")
    parser.add_argument("--y-column", default="y")
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument(
        "--max-agents",
        type=int,
        default=0,
        help="optional safety cap; 0 means no cap",
    )
    return parser.parse_args()


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_centroids(
    path: Path,
    taz_column: str,
    x_column: str,
    y_column: str,
) -> dict[str, tuple[float, float]]:
    centroids: dict[str, tuple[float, float]] = {}
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            taz = row.get(taz_column, "")
            if not taz:
                continue
            centroids[taz] = (
                parse_float(row.get(x_column, "")),
                parse_float(row.get(y_column, "")),
            )
    return centroids


def expansion_count(weight: float, rng: random.Random) -> int:
    base = int(math.floor(weight))
    return base + int(rng.random() < weight - base)


def minutes_to_time(value: str) -> str:
    minutes = int(round(parse_float(value)))
    minutes = max(0, minutes)
    hours = minutes // 60
    minute = minutes % 60
    return f"{hours:02d}:{minute:02d}:00"


def open_output(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def activity_type(value: str, fallback: str) -> str:
    value = value.strip()
    return f"act_{value}" if value else fallback


def write_person(
    output_file: TextIO,
    person_id: str,
    row: dict[str, str],
    origin: tuple[float, float],
    destination: tuple[float, float],
) -> None:
    origin_type = escape(activity_type(row.get("o_activity", ""), "origin"))
    destination_type = escape(activity_type(row.get("d_activity", ""), "destination"))
    departure_time = minutes_to_time(row.get("departure_time_min", "0"))
    output_file.write(f'  <person id="{escape(person_id)}">\n')
    output_file.write('    <plan selected="yes">\n')
    output_file.write(
        f'      <act type="{origin_type}" x="{origin[0]:.6f}" y="{origin[1]:.6f}" '
        f'end_time="{departure_time}" />\n'
    )
    output_file.write('      <leg mode="car" />\n')
    output_file.write(
        f'      <act type="{destination_type}" x="{destination[0]:.6f}" '
        f'y="{destination[1]:.6f}" />\n'
    )
    output_file.write('    </plan>\n')
    output_file.write('  </person>\n')


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    centroids = read_centroids(
        args.taz_centroids,
        args.taz_column,
        args.x_column,
        args.y_column,
    )

    written = 0
    skipped = 0
    with args.vehicle_trips.open(newline="") as input_file, open_output(args.out) as output_file:
        reader = csv.DictReader(input_file)
        output_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        output_file.write('<population>\n')
        for row in reader:
            origin = centroids.get(row.get("o_tpb_taz", ""))
            destination = centroids.get(row.get("d_tpb_taz", ""))
            if origin is None or destination is None:
                skipped += 1
                continue

            count = expansion_count(parse_float(row.get("vehicle_weight", "")), rng)
            for index in range(count):
                if args.max_agents and written >= args.max_agents:
                    break
                person_id = (
                    f"{row.get('scenario', 'scenario')}_"
                    f"{row.get('source_row', 'row')}_{index + 1}"
                )
                write_person(output_file, person_id, row, origin, destination)
                written += 1
            if args.max_agents and written >= args.max_agents:
                break
        output_file.write('</population>\n')

    print(f"wrote {args.out}")
    print(f"agents written: {written}")
    print(f"rows skipped for missing centroids: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
