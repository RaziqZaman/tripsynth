#!/usr/bin/env python3
"""Convert Stage 1 network_links.csv into a simple bidirectional MATSim network."""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from pathlib import Path
from typing import TextIO
from xml.sax.saxutils import escape


NETWORK_LINKS = Path("06x_stage1/network_links.csv")
OUTPUT_NETWORK = Path("06x_stage1/network.xml.gz")
OUTPUT_LINK_MAP = Path("06x_stage1/matsim_link_count_map.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-links", type=Path, default=NETWORK_LINKS)
    parser.add_argument("--out", type=Path, default=OUTPUT_NETWORK)
    parser.add_argument("--link-map-out", type=Path, default=OUTPUT_LINK_MAP)
    return parser.parse_args()


def parse_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def link_defaults(f_system: str, lanes: float) -> tuple[float, float]:
    try:
        code = int(float(f_system))
    except (TypeError, ValueError):
        code = 7

    if code in {1, 2}:
        freespeed = 29.06  # 65 mph
        capacity_per_lane = 2000.0
    elif code == 3:
        freespeed = 22.35  # 50 mph
        capacity_per_lane = 1600.0
    elif code == 4:
        freespeed = 17.88  # 40 mph
        capacity_per_lane = 1200.0
    elif code in {5, 6}:
        freespeed = 13.41  # 30 mph
        capacity_per_lane = 900.0
    else:
        freespeed = 11.18  # 25 mph
        capacity_per_lane = 600.0

    lanes = max(lanes, 1.0)
    return freespeed, capacity_per_lane * lanes


def node_key(lon: str, lat: str) -> tuple[str, str]:
    return f"{parse_float(lon, 0.0):.8f}", f"{parse_float(lat, 0.0):.8f}"


def read_network_links(path: Path) -> tuple[list[dict[str, str]], dict[tuple[str, str], str]]:
    csv.field_size_limit(sys.maxsize)
    links: list[dict[str, str]] = []
    node_ids: dict[tuple[str, str], str] = {}

    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            from_key = node_key(row.get("from_lon", ""), row.get("from_lat", ""))
            to_key = node_key(row.get("to_lon", ""), row.get("to_lat", ""))
            if from_key == to_key:
                continue
            if from_key not in node_ids:
                node_ids[from_key] = f"n{len(node_ids) + 1}"
            if to_key not in node_ids:
                node_ids[to_key] = f"n{len(node_ids) + 1}"
            links.append(row)
    return links, node_ids


def open_output(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def write_network(
    path: Path,
    links: list[dict[str, str]],
    node_ids: dict[tuple[str, str], str],
) -> list[dict[str, str]]:
    matsim_map: list[dict[str, str]] = []
    with open_output(path) as output_file:
        output_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        output_file.write('<!DOCTYPE network SYSTEM "http://www.matsim.org/files/dtd/network_v2.dtd">\n')
        output_file.write('<network name="mdot_aadt_stage1">\n')
        output_file.write('  <nodes>\n')
        for (lon, lat), node_id in sorted(node_ids.items(), key=lambda item: item[1]):
            output_file.write(
                f'    <node id="{escape(node_id)}" x="{lon}" y="{lat}" />\n'
            )
        output_file.write('  </nodes>\n')
        output_file.write('  <links capperiod="01:00:00" effectivecellsize="7.5" effectivelanewidth="3.75">\n')

        for row in links:
            count_link_id = row["link_id"]
            from_key = node_key(row.get("from_lon", ""), row.get("from_lat", ""))
            to_key = node_key(row.get("to_lon", ""), row.get("to_lat", ""))
            from_node = node_ids[from_key]
            to_node = node_ids[to_key]
            length = max(parse_float(row.get("length_m", ""), 1.0), 1.0)
            lanes = max(parse_float(row.get("lanes", ""), 1.0), 1.0)
            freespeed, capacity = link_defaults(row.get("f_system", ""), lanes)
            for suffix, link_from, link_to in [
                ("fwd", from_node, to_node),
                ("rev", to_node, from_node),
            ]:
                matsim_link_id = f"{count_link_id}_{suffix}"
                output_file.write(
                    f'    <link id="{escape(matsim_link_id)}" from="{escape(link_from)}" '
                    f'to="{escape(link_to)}" length="{length:.3f}" '
                    f'freespeed="{freespeed:.3f}" capacity="{capacity:.3f}" '
                    f'permlanes="{lanes:.3f}" modes="car" />\n'
                )
                matsim_map.append(
                    {
                        "matsim_link_id": matsim_link_id,
                        "count_link_id": count_link_id,
                        "direction": suffix,
                    }
                )

        output_file.write('  </links>\n')
        output_file.write('</network>\n')
    return matsim_map


def write_link_map(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["matsim_link_id", "count_link_id", "direction"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    links, node_ids = read_network_links(args.network_links)
    matsim_map = write_network(args.out, links, node_ids)
    write_link_map(args.link_map_out, matsim_map)
    print(f"wrote {args.out}")
    print(f"wrote {args.link_map_out}")
    print(f"nodes: {len(node_ids)}")
    print(f"count links: {len(links)}")
    print(f"matsim directed links: {len(matsim_map)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
