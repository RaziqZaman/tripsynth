"""Optional VDOT observed-count adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd

from tripsynth.config import resolve_path
from tripsynth.data_sources.observed_counts.arcgis import ArcGISFeatureServerClient
from tripsynth.data_sources.observed_counts.base import normalize_observed_counts
from tripsynth.data_sources.observed_counts.local_counts import _read_local_geodata


def load_vdot_counts(
    config: dict[str, Any],
    *,
    force: bool = False,
    client: ArcGISFeatureServerClient | None = None,
) -> gpd.GeoDataFrame:
    source = config.get("observed_counts", {}).get("sources", {}).get("vdot_official", {})
    field_map = source.get("field_map", {})
    if source.get("url"):
        client = client or ArcGISFeatureServerClient(force=force)
        raw, _, _ = client.download_layer(
            source["url"],
            layer=int(source.get("layer", 0)),
            cache_dir=Path("data/raw/observed_counts/vdot_official"),
        )
    else:
        path = resolve_path(config, source.get("local_path"))
        if path is None or not path.exists():
            raise FileNotFoundError(
                "VDOT adapter is enabled but no valid url or local_path was configured."
            )
        raw = _read_local_geodata(path, field_map)

    normalized = normalize_observed_counts(
        raw,
        source="vdot_official",
        state="va",
        field_map=field_map,
        temporal_type=source.get("temporal_type", "unknown"),
    )
    return normalized.gdf
