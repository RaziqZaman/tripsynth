#!/usr/bin/env python3
"""Download the Maryland Geofabrik OSM PBF used for the routable Stage 1 network."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm


DEFAULT_URL = "https://download.geofabrik.de/north-america/us/maryland-latest.osm.pbf"
DEFAULT_OUT = Path("06x_stage1_osm/maryland-latest.osm.pbf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="redownload even if the output file already exists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and not args.force:
        print(f"{args.out} already exists; use --force to redownload")
        print(f"size: {args.out.stat().st_size:,} bytes")
        return 0

    temp_path = args.out.with_suffix(args.out.suffix + ".part")
    with requests.get(args.url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", "0") or 0)
        with temp_path.open("wb") as output_file, tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="download OSM PBF",
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                output_file.write(chunk)
                progress.update(len(chunk))

    temp_path.replace(args.out)
    print(f"wrote {args.out}")
    print(f"size: {args.out.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
