#!/usr/bin/env python3
"""Download observed traffic-count attributes from an ArcGIS REST query layer."""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_QUERY_URL = (
    "https://mdgeodata.md.gov/imap/rest/services/Transportation/"
    "MD_AnnualAverageDailyTraffic/FeatureServer/0/query"
)
OUTPUT_CSV = Path("06x_stage1/observed_counts_raw.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-url", default=DEFAULT_QUERY_URL)
    parser.add_argument("--out", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--where", default="1=1")
    parser.add_argument("--out-fields", default="*")
    parser.add_argument("--out-sr", default="4326")
    parser.add_argument("--page-size", type=int, default=2000)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument(
        "--include-geometry-json",
        action="store_true",
        help="include compact geometry_json for line/polygon map matching",
    )
    return parser.parse_args()


def fetch_page(
    query_url: str,
    where: str,
    out_fields: str,
    out_sr: str,
    page_size: int,
    offset: int,
) -> dict[str, Any]:
    params = {
        "f": "json",
        "where": where,
        "outFields": out_fields,
        "returnGeometry": "true",
        "outSR": out_sr,
        "resultRecordCount": str(page_size),
        "resultOffset": str(offset),
    }
    url = f"{query_url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload


def geometry_fields(feature: dict[str, Any], include_geometry_json: bool) -> dict[str, Any]:
    geometry = feature.get("geometry") or {}
    output: dict[str, Any] = {}
    for key in ["x", "y", "longitude", "latitude"]:
        if key in geometry:
            output[f"geometry_{key}"] = geometry[key]
    if include_geometry_json and geometry:
        output["geometry_json"] = json.dumps(geometry, separators=(",", ":"))
    return output


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    features: list[dict[str, Any]] = []
    offset = 0
    while True:
        payload = fetch_page(
            args.query_url,
            args.where,
            args.out_fields,
            args.out_sr,
            args.page_size,
            offset,
        )
        page = payload.get("features", [])
        if not page:
            break
        features.extend(page)
        print(f"downloaded {len(features)} features")
        if len(page) < args.page_size and not payload.get("exceededTransferLimit"):
            break
        offset += len(page)
        time.sleep(args.sleep)

    rows: list[dict[str, Any]] = []
    fieldnames: set[str] = set()
    for feature in features:
        row = {
            **(feature.get("attributes") or {}),
            **geometry_fields(feature, args.include_geometry_json),
        }
        rows.append(row)
        fieldnames.update(row)

    with args.out.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=sorted(fieldnames))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
