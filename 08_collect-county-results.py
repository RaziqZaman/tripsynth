#!/usr/bin/env python3
"""Collect county-level MATSim validation summaries into one table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


OUT_ROOT = Path("08x_county_runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--out", type=Path, default=OUT_ROOT / "county_validation_summary.csv")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as input_file:
        return list(csv.DictReader(input_file))


def main() -> int:
    args = parse_args()
    rows = []
    for manifest_row in read_manifest(args.out_root / "county_manifest.csv"):
        county = manifest_row.get("county", "")
        slug = manifest_row.get("slug", "")
        county_dir = args.out_root / slug
        summary_path = county_dir / "count_validation_summary.csv"
        if not summary_path.exists():
            rows.append({
                "county": county,
                "slug": slug,
                "scenario": "",
                "status": "missing_summary",
                "observed_counts": manifest_row.get("observed_counts", ""),
                "network_base_links": manifest_row.get("network_base_links", ""),
                "real_vehicle_trips": manifest_row.get("real_vehicle_trips", ""),
                "synthetic_vehicle_trips": manifest_row.get("synthetic_vehicle_trips", ""),
            })
            continue
        with summary_path.open(newline="") as input_file:
            for row in csv.DictReader(input_file):
                rows.append({
                    "county": county,
                    "slug": slug,
                    "status": "completed",
                    "observed_counts": manifest_row.get("observed_counts", ""),
                    "network_base_links": manifest_row.get("network_base_links", ""),
                    "real_vehicle_trips": manifest_row.get("real_vehicle_trips", ""),
                    "synthetic_vehicle_trips": manifest_row.get("synthetic_vehicle_trips", ""),
                    **row,
                })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "county", "slug", "scenario", "status", "observed_counts", "network_base_links",
        "real_vehicle_trips", "synthetic_vehicle_trips", "matched_counts", "observed_total",
        "model_total", "mean_bias", "mae", "rmse", "mape", "correlation", "geh_lt_5_share",
    ]
    with args.out.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
