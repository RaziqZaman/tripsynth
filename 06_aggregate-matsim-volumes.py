#!/usr/bin/env python3
"""Aggregate directed MATSim link volumes back to observed count-link IDs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


LINK_MAP = Path("06x_stage1/matsim_link_count_map.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--link-volumes", type=Path, required=True)
    parser.add_argument("--link-map", type=Path, default=LINK_MAP)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--matsim-link-column", default="link_id")
    parser.add_argument("--volume-column", default="volume")
    return parser.parse_args()


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_link_map(path: Path) -> dict[str, str]:
    mapping = {}
    with path.open(newline="") as input_file:
        for row in csv.DictReader(input_file):
            mapping[row["matsim_link_id"]] = row["count_link_id"]
    return mapping


def main() -> int:
    args = parse_args()
    mapping = read_link_map(args.link_map)
    totals: dict[str, float] = {}
    unmatched = 0

    with args.link_volumes.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            matsim_link_id = row.get(args.matsim_link_column, "")
            count_link_id = mapping.get(matsim_link_id)
            if count_link_id is None:
                unmatched += 1
                continue
            totals[count_link_id] = totals.get(count_link_id, 0.0) + parse_float(
                row.get(args.volume_column, "")
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["link_id", "volume"])
        writer.writeheader()
        for link_id, volume in sorted(totals.items()):
            writer.writerow({"link_id": link_id, "volume": f"{volume:.8f}"})

    print(f"wrote {args.out}")
    print(f"aggregated count links: {len(totals)}")
    print(f"unmatched input links: {unmatched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
