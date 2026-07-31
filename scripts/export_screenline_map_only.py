#!/usr/bin/env python3
"""Export one saved screenline example as a clean cartographic panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from generate_screenline_examples import (
    DEFAULT_FHWA_HOURLY,
    DEFAULT_FHWA_METADATA,
    DEFAULT_RUN,
    ROOT,
    _load_fhwa_points,
    _plot_map,
    _resolve_station_assignments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=ROOT / "screenline_examples_exactly_2",
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--example-id", type=int, required=True)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_dir = args.batch_dir.resolve()
    geo_dir = args.run_dir.resolve() / "geo"

    manifest = pd.read_csv(
        batch_dir / "manifest.csv",
        dtype={"o_tract_fips": "string", "d_tract_fips": "string"},
    )
    selected = manifest[manifest["example_id"].eq(args.example_id)]
    if len(selected) != 1:
        raise ValueError(f"Expected one manifest row for example {args.example_id}; found {len(selected)}")
    manifest_row = selected.iloc[0]
    origin = str(manifest_row["o_tract_fips"]).zfill(11)
    destination = str(manifest_row["d_tract_fips"]).zfill(11)

    tracts = gpd.read_parquet(geo_dir / "tracts.parquet")
    tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)
    tract_lookup = tracts.set_index("GEOID", drop=False)
    adjacency = gpd.read_parquet(geo_dir / "tract_adjacency.parquet")
    adjacency["screenline_id"] = adjacency["screenline_id"].astype(str)
    geometry_lookup = adjacency.set_index("screenline_id")["geometry"]

    saved_screenlines = pd.read_csv(
        batch_dir / "screenlines.csv",
        dtype={
            "o_tract_fips": "string",
            "d_tract_fips": "string",
            "screenline_id": "string",
            "tract_a": "string",
            "tract_b": "string",
        },
    )
    path_rows = (
        saved_screenlines[saved_screenlines["example_id"].eq(args.example_id)]
        .sort_values("path_order")
        .copy()
    )
    if path_rows.empty:
        raise ValueError(f"No saved screenlines for example {args.example_id}")
    path_rows["geometry"] = path_rows["screenline_id"].map(geometry_lookup)
    if path_rows["geometry"].isna().any():
        raise ValueError("One or more saved screenlines lack adjacency geometry")

    station_map = pd.read_parquet(geo_dir / "screenline_station_map.parquet")
    station_map["screenline_id"] = station_map["screenline_id"].astype(str)
    station_map["station_id"] = station_map["station_id"].astype(str)
    aadt_points = gpd.read_parquet(geo_dir / "aadt_points.parquet")
    aadt_points["station_id"] = aadt_points["station_id"].astype(str)
    fhwa_points = _load_fhwa_points(DEFAULT_FHWA_METADATA, DEFAULT_FHWA_HOURLY, tracts.crs)
    fhwa_ids = set(fhwa_points["station_id"].astype(str))
    assignments = _resolve_station_assignments(path_rows, station_map, aadt_points, fhwa_ids)

    origin_row = tract_lookup.loc[origin]
    destination_row = tract_lookup.loc[destination]
    od_line = LineString(
        [
            (float(origin_row["rep_x"]), float(origin_row["rep_y"])),
            (float(destination_row["rep_x"]), float(destination_row["rep_y"])),
        ]
    )
    ordered_tract_ids = {origin, destination}
    ordered_tract_ids.update(path_rows["tract_a"].astype(str))
    ordered_tract_ids.update(path_rows["tract_b"].astype(str))

    if args.output is None:
        output_dir = batch_dir / "map_only"
        output_dir.mkdir(parents=True, exist_ok=True)
        original_stem = Path(str(manifest_row["filename"])).stem
        output_path = output_dir / f"{original_stem}_map_only.png"
    else:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    _plot_map(
        output_path,
        int(args.example_id),
        origin,
        destination,
        od_line,
        path_rows,
        ordered_tract_ids,
        tracts,
        tract_lookup,
        aadt_points,
        fhwa_points,
        assignments,
        int(args.dpi),
        "#FFFFFF",
        map_only=True,
    )
    print(output_path)


if __name__ == "__main__":
    main()
