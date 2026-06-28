"""Generic local observed-count file adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import Point

from tripsynth.config import resolve_path
from tripsynth.data_sources.observed_counts.base import (
    ObservedCountsError,
    normalize_observed_counts,
)


def _read_local_geodata(path: Path, field_map: dict[str, str | None]) -> gpd.GeoDataFrame:
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith((".gpkg", ".geojson", ".json", ".shp", ".kml")):
        return gpd.read_file(path)
    if suffix.endswith((".parquet", ".geoparquet")):
        return gpd.read_parquet(path)
    if suffix.endswith(".csv"):
        df = pd.read_csv(path)
        geometry_field = field_map.get("geometry") or field_map.get("wkt")
        lon_field = field_map.get("lon") or field_map.get("longitude")
        lat_field = field_map.get("lat") or field_map.get("latitude")
        if geometry_field and geometry_field in df:
            geometry = df[geometry_field].apply(lambda value: wkt.loads(value) if pd.notna(value) else None)
        elif lon_field and lat_field and lon_field in df and lat_field in df:
            geometry = [
                Point(xy) if pd.notna(xy[0]) and pd.notna(xy[1]) else None
                for xy in zip(df[lon_field], df[lat_field])
            ]
        else:
            raise ObservedCountsError(
                "CSV observed counts require field_map.geometry/wkt or lat/lon fields."
            )
        return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    raise ObservedCountsError(f"Unsupported observed-count local file format: {path}")


def load_local_observed_counts(
    config: dict[str, Any],
    *,
    source_name: str = "local_user_counts",
) -> gpd.GeoDataFrame:
    source = config.get("observed_counts", {}).get("sources", {}).get(source_name, {})
    path = resolve_path(config, source.get("path") or source.get("local_path"))
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"Observed count file for {source_name} was not found. Configure local_path/path."
        )
    field_map = source.get("field_map", {})
    raw = _read_local_geodata(path, field_map)
    normalized = normalize_observed_counts(
        raw,
        source=source_name,
        state=source.get("state", "unknown"),
        field_map=field_map,
        temporal_type=source.get("temporal_type", "unknown"),
    )
    return normalized.gdf
