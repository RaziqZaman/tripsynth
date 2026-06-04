#!/usr/bin/env python3
"""Match observed travel-time segment speeds to MATSim OSM links."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_SPEEDS = Path("06x_stage1_osm/observed_speeds_raw.csv")
DEFAULT_LINKS = Path("06x_stage1_osm/network_links.csv")
DEFAULT_OUT = Path("06x_stage1_osm/observed_speeds.csv")
EARTH_M_PER_DEG = 111_320.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speeds", type=Path, default=DEFAULT_SPEEDS)
    parser.add_argument("--links", type=Path, default=DEFAULT_LINKS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-distance-m", type=float, default=500.0)
    parser.add_argument("--cell-size-m", type=float, default=2_000.0)
    parser.add_argument("--route-match", choices=["preferred", "required", "off"], default="preferred")
    return parser.parse_args()


def parse_float(value: str | None, default: float = math.nan) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_route_tokens(*values: str) -> set[str]:
    text = " ".join(value or "" for value in values).upper()
    text = text.replace("INTERSTATE", " I ")
    text = text.replace("U.S.", " US ").replace("US ROUTE", " US ")
    text = text.replace("UNITED STATES", " US ")
    text = text.replace("STATE ROUTE", " SR ").replace("ROUTE", " SR ")
    text = text.replace("VIRGINIA", " VA ")
    tokens: set[str] = set()
    for prefix, number in re.findall(r"\b(I|US|VA|SR)\s*[- ]?\s*(\d+[A-Z]?)\b", text):
        canonical_prefix = "VA" if prefix == "SR" else prefix
        tokens.add(f"{canonical_prefix}{number}")
    for number in re.findall(r"\b(?:I|US|VA|SR)?\s*-?\s*(\d{1,4}[A-Z]?)\b", text):
        tokens.add(number)
    return tokens


def latlon_to_xy(lon: float, lat: float, lat0: float) -> tuple[float, float]:
    scale_x = EARTH_M_PER_DEG * math.cos(math.radians(lat0))
    return lon * scale_x, lat * EARTH_M_PER_DEG


def point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    qx = ax + t * dx
    qy = ay + t * dy
    return math.hypot(px - qx, py - qy)


def segment_distance_m(obs: dict[str, float], link: dict[str, object]) -> float:
    lat0 = (obs["start_lat"] + obs["end_lat"] + float(link["from_lat"]) + float(link["to_lat"])) / 4.0
    ox1, oy1 = latlon_to_xy(obs["start_lon"], obs["start_lat"], lat0)
    ox2, oy2 = latlon_to_xy(obs["end_lon"], obs["end_lat"], lat0)
    lx1, ly1 = latlon_to_xy(float(link["from_lon"]), float(link["from_lat"]), lat0)
    lx2, ly2 = latlon_to_xy(float(link["to_lon"]), float(link["to_lat"]), lat0)
    return min(
        point_segment_distance(ox1, oy1, lx1, ly1, lx2, ly2),
        point_segment_distance(ox2, oy2, lx1, ly1, lx2, ly2),
        point_segment_distance(lx1, ly1, ox1, oy1, ox2, oy2),
        point_segment_distance(lx2, ly2, ox1, oy1, ox2, oy2),
    )


def direction_for_match(obs: dict[str, float], link: dict[str, object]) -> str:
    lat0 = (obs["start_lat"] + obs["end_lat"] + float(link["from_lat"]) + float(link["to_lat"])) / 4.0
    ox1, oy1 = latlon_to_xy(obs["start_lon"], obs["start_lat"], lat0)
    ox2, oy2 = latlon_to_xy(obs["end_lon"], obs["end_lat"], lat0)
    lx1, ly1 = latlon_to_xy(float(link["from_lon"]), float(link["from_lat"]), lat0)
    lx2, ly2 = latlon_to_xy(float(link["to_lon"]), float(link["to_lat"]), lat0)
    dot = (ox2 - ox1) * (lx2 - lx1) + (oy2 - oy1) * (ly2 - ly1)
    return "fwd" if dot >= 0 else "rev"


def observed_geometry(row: dict[str, str]) -> dict[str, float] | None:
    coords = {
        "start_lon": parse_float(row.get("start_lon")),
        "start_lat": parse_float(row.get("start_lat")),
        "end_lon": parse_float(row.get("end_lon")),
        "end_lat": parse_float(row.get("end_lat")),
    }
    if any(math.isnan(value) for value in coords.values()):
        return None
    if coords["start_lon"] == coords["end_lon"] and coords["start_lat"] == coords["end_lat"]:
        return None
    return coords


def read_links(path: Path, cell_size_m: float) -> tuple[list[dict[str, object]], dict[tuple[int, int], list[int]], float, float]:
    links: list[dict[str, object]] = []
    lats: list[float] = []
    with path.open(newline="") as input_file:
        for row in csv.DictReader(input_file):
            from_lon = parse_float(row.get("from_lon"))
            from_lat = parse_float(row.get("from_lat"))
            to_lon = parse_float(row.get("to_lon"))
            to_lat = parse_float(row.get("to_lat"))
            if any(math.isnan(value) for value in [from_lon, from_lat, to_lon, to_lat]):
                continue
            row_obj: dict[str, object] = {
                "link_id": row.get("link_id", ""),
                "from_lon": from_lon,
                "from_lat": from_lat,
                "to_lon": to_lon,
                "to_lat": to_lat,
                "length_m": parse_float(row.get("length_m"), 0.0),
                "highway": row.get("highway", ""),
                "name": row.get("name", ""),
                "ref": row.get("ref", ""),
                "route_ref": row.get("route_ref", ""),
                "route_tokens": normalize_route_tokens(row.get("ref", ""), row.get("route_ref", ""), row.get("name", "")),
            }
            links.append(row_obj)
            lats.extend([from_lat, to_lat])
    lat0 = sum(lats) / len(lats) if lats else 38.0
    cell_deg_lat = cell_size_m / EARTH_M_PER_DEG
    cell_deg_lon = cell_size_m / (EARTH_M_PER_DEG * max(0.1, math.cos(math.radians(lat0))))
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, link in enumerate(links):
        lon_mid = (float(link["from_lon"]) + float(link["to_lon"])) / 2.0
        lat_mid = (float(link["from_lat"]) + float(link["to_lat"])) / 2.0
        grid[(int(lon_mid // cell_deg_lon), int(lat_mid // cell_deg_lat))].append(index)
    return links, grid, cell_deg_lon, cell_deg_lat


def candidate_indices(
    grid: dict[tuple[int, int], list[int]],
    obs: dict[str, float],
    cell_deg_lon: float,
    cell_deg_lat: float,
    max_distance_m: float,
) -> set[int]:
    expand_lat = max_distance_m / EARTH_M_PER_DEG
    mid_lat = (obs["start_lat"] + obs["end_lat"]) / 2.0
    expand_lon = max_distance_m / (EARTH_M_PER_DEG * max(0.1, math.cos(math.radians(mid_lat))))
    lon_min = min(obs["start_lon"], obs["end_lon"]) - expand_lon
    lon_max = max(obs["start_lon"], obs["end_lon"]) + expand_lon
    lat_min = min(obs["start_lat"], obs["end_lat"]) - expand_lat
    lat_max = max(obs["start_lat"], obs["end_lat"]) + expand_lat
    ix_min = int(lon_min // cell_deg_lon)
    ix_max = int(lon_max // cell_deg_lon)
    iy_min = int(lat_min // cell_deg_lat)
    iy_max = int(lat_max // cell_deg_lat)
    found: set[int] = set()
    for ix in range(ix_min, ix_max + 1):
        for iy in range(iy_min, iy_max + 1):
            found.update(grid.get((ix, iy), []))
    return found


def best_match(
    row: dict[str, str],
    obs: dict[str, float],
    links: list[dict[str, object]],
    candidates: set[int],
    max_distance_m: float,
    route_match: str,
) -> tuple[dict[str, object], float] | None:
    obs_tokens = normalize_route_tokens(row.get("route", ""), row.get("description", ""))
    best: tuple[dict[str, object], float] | None = None
    route_candidates = []
    spatial_candidates = []
    for index in candidates:
        link = links[index]
        link_tokens = link.get("route_tokens", set())
        route_ok = bool(obs_tokens and link_tokens and obs_tokens.intersection(link_tokens))
        if route_match == "required" and not route_ok:
            continue
        distance = segment_distance_m(obs, link)
        if distance <= max_distance_m:
            (route_candidates if route_ok else spatial_candidates).append((link, distance))
    pool = route_candidates if route_candidates or route_match in {"preferred", "required"} else spatial_candidates
    if route_match == "preferred" and not route_candidates:
        pool = spatial_candidates
    for link, distance in pool:
        if best is None or distance < best[1]:
            best = (link, distance)
    return best


def main() -> int:
    args = parse_args()
    links, grid, cell_deg_lon, cell_deg_lat = read_links(args.links, args.cell_size_m)
    rows: list[dict[str, str]] = []
    skipped_missing_geometry = 0
    skipped_no_match = 0
    with args.speeds.open(newline="") as input_file:
        for row in csv.DictReader(input_file):
            obs = observed_geometry(row)
            if obs is None:
                skipped_missing_geometry += 1
                continue
            candidates = candidate_indices(grid, obs, cell_deg_lon, cell_deg_lat, args.max_distance_m)
            match = best_match(row, obs, links, candidates, args.max_distance_m, args.route_match)
            if match is None:
                skipped_no_match += 1
                continue
            link, distance = match
            direction = direction_for_match(obs, link)
            rows.append(
                {
                    "segment_id": row.get("segment_id", ""),
                    "segment_group": row.get("segment_group", ""),
                    "matsim_link_id": f"{link['link_id']}_{direction}",
                    "link_id": str(link["link_id"]),
                    "observed_direction": row.get("direction", ""),
                    "matched_direction": direction,
                    "observed_speed_mph": row.get("observed_speed_mph", ""),
                    "observed_length_miles": row.get("length_miles", ""),
                    "observed_travel_time_min": row.get("travel_time_min", ""),
                    "observed_status": row.get("status", ""),
                    "congestion": row.get("congestion", ""),
                    "imputation": row.get("imputation", ""),
                    "route": row.get("route", ""),
                    "description": row.get("description", ""),
                    "start_lon": row.get("start_lon", ""),
                    "start_lat": row.get("start_lat", ""),
                    "end_lon": row.get("end_lon", ""),
                    "end_lat": row.get("end_lat", ""),
                    "match_distance_m": f"{distance:.8f}",
                    "link_length_m": f"{float(link['length_m']):.8f}",
                    "link_highway": str(link.get("highway", "")),
                    "link_ref": str(link.get("ref", "")),
                    "link_name": str(link.get("name", "")),
                    "observed_source": row.get("source_response_url", ""),
                    "fetched_at_utc": row.get("fetched_at_utc", ""),
                }
            )

    fieldnames = [
        "segment_id", "segment_group", "matsim_link_id", "link_id", "observed_direction", "matched_direction",
        "observed_speed_mph", "observed_length_miles", "observed_travel_time_min", "observed_status",
        "congestion", "imputation", "route", "description", "start_lon", "start_lat", "end_lon", "end_lat",
        "match_distance_m", "link_length_m", "link_highway", "link_ref", "link_name", "observed_source", "fetched_at_utc",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"network links read: {len(links):,}")
    print(f"observed speed segments matched: {len(rows):,}")
    print(f"skipped missing geometry: {skipped_missing_geometry:,}")
    print(f"skipped no match within {args.max_distance_m:g} m: {skipped_no_match:,}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
