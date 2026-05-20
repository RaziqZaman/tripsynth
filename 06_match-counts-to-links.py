#!/usr/bin/env python3
"""Nearest-link match observed count points to a road-network links CSV."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


COUNTS_CSV = Path("06x_stage1/observed_counts_points.csv")
OUTPUT_CSV = Path("06x_stage1/observed_counts.csv")
EARTH_RADIUS_M = 6_371_000.0
REFERENCE_LAT = 39.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", type=Path, default=COUNTS_CSV)
    parser.add_argument("--links", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--count-x-column", default="lon")
    parser.add_argument("--count-y-column", default="lat")
    parser.add_argument("--link-id-column", default="link_id")
    parser.add_argument("--from-x-column", default="from_lon")
    parser.add_argument("--from-y-column", default="from_lat")
    parser.add_argument("--to-x-column", default="to_lon")
    parser.add_argument("--to-y-column", default="to_lat")
    parser.add_argument("--max-distance-m", type=float, default=100.0)
    parser.add_argument(
        "--cell-size-m",
        type=float,
        default=250.0,
        help="spatial index cell size; larger values use more candidates per count",
    )
    return parser.parse_args()


def parse_float(value: str | float, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def local_xy(lon: float, lat: float, origin_lon: float, origin_lat: float) -> tuple[float, float]:
    origin_lat_rad = math.radians(origin_lat)
    x = math.radians(lon - origin_lon) * math.cos(origin_lat_rad) * EARTH_RADIUS_M
    y = math.radians(lat - origin_lat) * EARTH_RADIUS_M
    return x, y


def distance_point_to_segment(
    point_lon: float,
    point_lat: float,
    from_lon: float,
    from_lat: float,
    to_lon: float,
    to_lat: float,
) -> float:
    px, py = local_xy(point_lon, point_lat, point_lon, point_lat)
    ax, ay = local_xy(from_lon, from_lat, point_lon, point_lat)
    bx, by = local_xy(to_lon, to_lat, point_lon, point_lat)
    abx = bx - ax
    aby = by - ay
    length2 = abx * abx + aby * aby
    if length2 == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / length2))
    closest_x = ax + t * abx
    closest_y = ay + t * aby
    return math.hypot(px - closest_x, py - closest_y)


def read_links(args: argparse.Namespace) -> list[dict[str, str | float]]:
    links: list[dict[str, str | float]] = []
    with args.links.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        for row in tqdm(reader, desc="read network links", unit="links"):
            link_id = row.get(args.link_id_column, "")
            if not link_id:
                continue
            link = dict(row)
            link["_from_lon"] = parse_float(row.get(args.from_x_column, ""))
            link["_from_lat"] = parse_float(row.get(args.from_y_column, ""))
            link["_to_lon"] = parse_float(row.get(args.to_x_column, ""))
            link["_to_lat"] = parse_float(row.get(args.to_y_column, ""))
            link["_link_id"] = link_id
            links.append(link)
    return links


def grid_steps(cell_size_m: float) -> tuple[float, float]:
    lat_step = cell_size_m / 111_320.0
    lon_step = cell_size_m / (
        111_320.0 * max(math.cos(math.radians(REFERENCE_LAT)), 0.1)
    )
    return lon_step, lat_step


def cell_range(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    lon_step: float,
    lat_step: float,
) -> tuple[range, range]:
    min_x = math.floor(min_lon / lon_step)
    max_x = math.floor(max_lon / lon_step)
    min_y = math.floor(min_lat / lat_step)
    max_y = math.floor(max_lat / lat_step)
    return range(min_x, max_x + 1), range(min_y, max_y + 1)


def lon_expand_degrees(distance_m: float, lat: float) -> float:
    return distance_m / (111_320.0 * max(math.cos(math.radians(lat)), 0.1))


def build_link_index(
    links: list[dict[str, str | float]],
    args: argparse.Namespace,
) -> tuple[dict[tuple[int, int], list[int]], float, float]:
    lon_step, lat_step = grid_steps(args.cell_size_m)
    index: dict[tuple[int, int], list[int]] = defaultdict(list)
    lat_expand = args.max_distance_m / 111_320.0

    for link_index, link in tqdm(
        enumerate(links),
        total=len(links),
        desc="index network links",
        unit="links",
    ):
        from_lon = float(link["_from_lon"])
        from_lat = float(link["_from_lat"])
        to_lon = float(link["_to_lon"])
        to_lat = float(link["_to_lat"])
        mid_lat = (from_lat + to_lat) / 2.0
        lon_expand = lon_expand_degrees(args.max_distance_m, mid_lat)
        x_cells, y_cells = cell_range(
            min(from_lon, to_lon) - lon_expand,
            min(from_lat, to_lat) - lat_expand,
            max(from_lon, to_lon) + lon_expand,
            max(from_lat, to_lat) + lat_expand,
            lon_step,
            lat_step,
        )
        for x_cell in x_cells:
            for y_cell in y_cells:
                index[(x_cell, y_cell)].append(link_index)
    return index, lon_step, lat_step


def candidate_link_indexes(
    count: dict[str, str],
    index: dict[tuple[int, int], list[int]],
    lon_step: float,
    lat_step: float,
    args: argparse.Namespace,
) -> set[int]:
    point_lon = parse_float(count.get(args.count_x_column, ""))
    point_lat = parse_float(count.get(args.count_y_column, ""))
    lat_expand = args.max_distance_m / 111_320.0
    lon_expand = lon_expand_degrees(args.max_distance_m, point_lat)
    x_cells, y_cells = cell_range(
        point_lon - lon_expand,
        point_lat - lat_expand,
        point_lon + lon_expand,
        point_lat + lat_expand,
        lon_step,
        lat_step,
    )
    candidates: set[int] = set()
    for x_cell in x_cells:
        for y_cell in y_cells:
            candidates.update(index.get((x_cell, y_cell), []))
    return candidates


def best_link(
    count: dict[str, str],
    links: list[dict[str, str | float]],
    candidates: set[int],
    args: argparse.Namespace,
) -> tuple[dict[str, str | float] | None, float]:
    point_lon = parse_float(count.get(args.count_x_column, ""))
    point_lat = parse_float(count.get(args.count_y_column, ""))
    best: dict[str, str | float] | None = None
    best_distance = float("inf")
    for link_index in candidates:
        link = links[link_index]
        distance = distance_point_to_segment(
            point_lon,
            point_lat,
            float(link["_from_lon"]),
            float(link["_from_lat"]),
            float(link["_to_lon"]),
            float(link["_to_lat"]),
        )
        if distance < best_distance:
            best = link
            best_distance = distance
    return best, best_distance


def main() -> int:
    args = parse_args()
    links = read_links(args)
    if not links:
        raise ValueError(f"no links read from {args.links}")
    index, lon_step, lat_step = build_link_index(links, args)

    matched = []
    unmatched = 0
    with args.counts.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        for count in tqdm(reader, desc="match count points", unit="counts"):
            candidates = candidate_link_indexes(count, index, lon_step, lat_step, args)
            link, distance = best_link(count, links, candidates, args)
            if link is None or distance > args.max_distance_m:
                unmatched += 1
                continue
            matched.append(
                {
                    "link_id": link["_link_id"],
                    "observed_volume": count.get("observed_volume", ""),
                    "count_id": count.get("count_id", ""),
                    "match_distance_m": f"{distance:.3f}",
                    "lon": count.get(args.count_x_column, ""),
                    "lat": count.get(args.count_y_column, ""),
                    "road_name": count.get("road_name", ""),
                    "route_id": count.get("route_id", ""),
                    "county": count.get("county", ""),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "link_id",
                "observed_volume",
                "count_id",
                "match_distance_m",
                "lon",
                "lat",
                "road_name",
                "route_id",
                "county",
            ],
        )
        writer.writeheader()
        writer.writerows(matched)

    print(f"wrote {args.out}")
    print(f"matched counts: {len(matched)}")
    print(f"unmatched counts: {unmatched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
