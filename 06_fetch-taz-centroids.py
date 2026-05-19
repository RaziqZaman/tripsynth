#!/usr/bin/env python3
"""Fetch TPB TAZ polygons and write centroid coordinates for Stage 1 demand."""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_QUERY_URL = "https://gis.mwcog.org/wa/rest/services/RTDC/TAZ/MapServer/1/query"
OUTPUT_CSV = Path("06x_stage1/taz_centroids.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-url", default=DEFAULT_QUERY_URL)
    parser.add_argument("--out", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--where", default="1=1")
    parser.add_argument("--out-fields", default="TAZ,STATE,STFIPS,CNTYFIPS,FIPSSTCO,REGION")
    parser.add_argument("--taz-field", default="TAZ")
    parser.add_argument("--out-sr", default="4326")
    parser.add_argument("--page-size", type=int, default=2000)
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
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload


def ring_centroid(ring: list[list[float]]) -> tuple[float, float, float]:
    area2 = 0.0
    cx6 = 0.0
    cy6 = 0.0
    for index in range(len(ring) - 1):
        x0, y0 = ring[index][:2]
        x1, y1 = ring[index + 1][:2]
        cross = x0 * y1 - x1 * y0
        area2 += cross
        cx6 += (x0 + x1) * cross
        cy6 += (y0 + y1) * cross
    if area2 == 0:
        xs = [point[0] for point in ring]
        ys = [point[1] for point in ring]
        return 0.0, sum(xs) / len(xs), sum(ys) / len(ys)
    return area2 / 2.0, cx6 / (3.0 * area2), cy6 / (3.0 * area2)


def polygon_centroid(geometry: dict[str, Any]) -> tuple[float, float]:
    rings = geometry.get("rings") or []
    if not rings:
        raise ValueError("feature has no polygon rings")

    weighted_x = 0.0
    weighted_y = 0.0
    total_area = 0.0
    fallback_points: list[list[float]] = []
    for ring in rings:
        fallback_points.extend(ring)
        area, cx, cy = ring_centroid(ring)
        weight = abs(area)
        weighted_x += cx * weight
        weighted_y += cy * weight
        total_area += weight

    if total_area == 0:
        xs = [point[0] for point in fallback_points]
        ys = [point[1] for point in fallback_points]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    return weighted_x / total_area, weighted_y / total_area


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    features: list[dict[str, Any]] = []
    offset = 0
    while True:
        payload = fetch_page(args, offset)
        page = payload.get("features", [])
        if not page:
            break
        features.extend(page)
        print(f"downloaded {len(features)} TAZ features")
        if len(page) < args.page_size and not payload.get("exceededTransferLimit"):
            break
        offset += len(page)
        time.sleep(args.sleep)

    rows = []
    for feature in features:
        attributes = feature.get("attributes") or {}
        taz = str(attributes.get(args.taz_field, "")).strip()
        if not taz:
            continue
        x, y = polygon_centroid(feature.get("geometry") or {})
        rows.append(
            {
                "taz": taz,
                "x": f"{x:.8f}",
                "y": f"{y:.8f}",
                "lon": f"{x:.8f}" if args.out_sr == "4326" else "",
                "lat": f"{y:.8f}" if args.out_sr == "4326" else "",
                "out_sr": args.out_sr,
                "state": str(attributes.get("STATE", "")),
                "stfips": str(attributes.get("STFIPS", "")),
                "cntyfips": str(attributes.get("CNTYFIPS", "")),
                "fipsstco": str(attributes.get("FIPSSTCO", "")),
                "region": str(attributes.get("REGION", "")),
            }
        )

    rows.sort(key=lambda row: int(row["taz"]) if row["taz"].isdigit() else row["taz"])
    with args.out.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["taz", "x", "y", "lon", "lat", "out_sr", "state", "stfips", "cntyfips", "fipsstco", "region"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.out}")
    print(f"TAZ centroids: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
