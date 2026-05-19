#!/usr/bin/env python3
"""Nearest-link match observed count points to a road-network links CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


COUNTS_CSV = Path("06x_stage1/observed_counts_points.csv")
OUTPUT_CSV = Path("06x_stage1/observed_counts.csv")
EARTH_RADIUS_M = 6_371_000.0


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
    return parser.parse_args()


def parse_float(value: str, default: float = 0.0) -> float:
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


def read_links(args: argparse.Namespace) -> list[dict[str, str]]:
    links = []
    with args.links.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            link_id = row.get(args.link_id_column, "")
            if not link_id:
                continue
            row["_from_lon"] = str(parse_float(row.get(args.from_x_column, "")))
            row["_from_lat"] = str(parse_float(row.get(args.from_y_column, "")))
            row["_to_lon"] = str(parse_float(row.get(args.to_x_column, "")))
            row["_to_lat"] = str(parse_float(row.get(args.to_y_column, "")))
            row["_link_id"] = link_id
            links.append(row)
    return links


def best_link(
    count: dict[str, str],
    links: list[dict[str, str]],
    args: argparse.Namespace,
) -> tuple[dict[str, str] | None, float]:
    point_lon = parse_float(count.get(args.count_x_column, ""))
    point_lat = parse_float(count.get(args.count_y_column, ""))
    best: dict[str, str] | None = None
    best_distance = float("inf")
    for link in links:
        distance = distance_point_to_segment(
            point_lon,
            point_lat,
            parse_float(link["_from_lon"]),
            parse_float(link["_from_lat"]),
            parse_float(link["_to_lon"]),
            parse_float(link["_to_lat"]),
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

    matched = []
    unmatched = 0
    with args.counts.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        for count in reader:
            link, distance = best_link(count, links, args)
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
