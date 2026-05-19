#!/usr/bin/env python3
"""Fetch MDOT AADT line features as count-bearing Stage 1 network links."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_QUERY_URL = (
    "https://mdgeodata.md.gov/imap/rest/services/Transportation/"
    "MD_AnnualAverageDailyTraffic/FeatureServer/1/query"
)
OUTPUT_DIR = Path("06x_stage1")
EARTH_RADIUS_M = 6_371_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-url", default=DEFAULT_QUERY_URL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--where", default="AADT > 0")
    parser.add_argument("--out-fields", default="*")
    parser.add_argument("--out-sr", default="4326")
    parser.add_argument("--page-size", type=int, default=20000)
    parser.add_argument("--sleep", type=float, default=0.1)
    return parser.parse_args()


def fetch_page(args: argparse.Namespace, offset: int) -> dict[str, Any]:
    params = {
        "f": "json",
        "where": args.where,
        "outFields": args.out_fields,
        "returnGeometry": "true",
        "outSR": args.out_sr,
        "resultRecordCount": str(args.page_size),
        "resultOffset": str(offset),
    }
    url = f"{args.query_url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=180) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def haversine_m(lon_a: float, lat_a: float, lon_b: float, lat_b: float) -> float:
    phi_a = math.radians(lat_a)
    phi_b = math.radians(lat_b)
    dphi = math.radians(lat_b - lat_a)
    dlambda = math.radians(lon_b - lon_a)
    h = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi_a) * math.cos(phi_b) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def line_length_m(paths: list[list[list[float]]]) -> float:
    length = 0.0
    for path in paths:
        for index in range(len(path) - 1):
            lon_a, lat_a = path[index][:2]
            lon_b, lat_b = path[index + 1][:2]
            length += haversine_m(lon_a, lat_a, lon_b, lat_b)
    return length


def first_last_point(paths: list[list[list[float]]]) -> tuple[list[float], list[float]]:
    nonempty = [path for path in paths if path]
    if not nonempty:
        raise ValueError("polyline has no points")
    return nonempty[0][0], nonempty[-1][-1]


def link_id(attributes: dict[str, Any]) -> str:
    location_id = str(attributes.get("LOCATION_ID") or "").strip()
    object_id = str(attributes.get("OBJECTID") or "").strip()
    if location_id and object_id:
        return f"mdot_aadt_{location_id}_{object_id}"
    if location_id:
        return f"mdot_aadt_{location_id}"
    return f"mdot_aadt_object_{object_id}"


def raw_row(feature: dict[str, Any]) -> dict[str, Any]:
    attributes = feature.get("attributes") or {}
    geometry = feature.get("geometry") or {}
    return {
        **attributes,
        "geometry_json": json.dumps(geometry, separators=(",", ":")),
    }


def output_rows(features: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    network_rows = []
    observed_rows = []
    for feature in features:
        attributes = feature.get("attributes") or {}
        geometry = feature.get("geometry") or {}
        paths = geometry.get("paths") or []
        if not paths:
            continue
        try:
            first, last = first_last_point(paths)
        except ValueError:
            continue

        observed_volume = parse_float(attributes.get("AADT"))
        length_m = line_length_m(paths)
        if observed_volume <= 0 or length_m <= 0 or first[:2] == last[:2]:
            continue

        current_link_id = link_id(attributes)
        network_rows.append(
            {
                "link_id": current_link_id,
                "from_lon": f"{first[0]:.8f}",
                "from_lat": f"{first[1]:.8f}",
                "to_lon": f"{last[0]:.8f}",
                "to_lat": f"{last[1]:.8f}",
                "length_m": f"{length_m:.3f}",
                "lanes": f"{parse_float(attributes.get('NUM_LANES'), 1.0):.3f}",
                "f_system": str(attributes.get("F_SYSTEM") or ""),
                "f_system_desc": str(attributes.get("F_SYSTEM_DESC") or ""),
                "route_id": str(attributes.get("ROUTEID") or ""),
                "road_name": str(attributes.get("ROADNAME") or ""),
                "county": str(attributes.get("COUNTY_DESC") or ""),
                "location_id": str(attributes.get("LOCATION_ID") or ""),
                "object_id": str(attributes.get("OBJECTID") or ""),
                "geometry_json": json.dumps(geometry, separators=(",", ":")),
            }
        )
        observed_rows.append(
            {
                "link_id": current_link_id,
                "observed_volume": f"{observed_volume:.8f}",
                "count_id": str(attributes.get("LOCATION_ID") or attributes.get("OBJECTID") or ""),
                "road_name": str(attributes.get("ROADNAME") or ""),
                "route_id": str(attributes.get("ROUTEID") or ""),
                "county": str(attributes.get("COUNTY_DESC") or ""),
            }
        )
    return network_rows, observed_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features: list[dict[str, Any]] = []
    offset = 0
    while True:
        payload = fetch_page(args, offset)
        page = payload.get("features", [])
        if not page:
            break
        features.extend(page)
        print(f"downloaded {len(features)} AADT line features")
        if len(page) < args.page_size and not payload.get("exceededTransferLimit"):
            break
        offset += len(page)
        time.sleep(args.sleep)

    raw_rows = [raw_row(feature) for feature in features]
    network_rows, observed_rows = output_rows(features)

    write_csv(args.output_dir / "mdot_aadt_lines_raw.csv", raw_rows)
    write_csv(args.output_dir / "network_links.csv", network_rows)
    write_csv(args.output_dir / "observed_counts.csv", observed_rows)

    print(f"wrote {args.output_dir / 'mdot_aadt_lines_raw.csv'}")
    print(f"wrote {args.output_dir / 'network_links.csv'}")
    print(f"wrote {args.output_dir / 'observed_counts.csv'}")
    print(f"network links: {len(network_rows)}")
    print(f"observed counts: {len(observed_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
