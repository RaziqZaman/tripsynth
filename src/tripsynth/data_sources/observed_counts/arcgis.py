"""ArcGIS FeatureServer downloader with cache-aware pagination."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon

from tripsynth.data_sources.observed_counts.base import ObservedCountsError

LOGGER = logging.getLogger(__name__)


class ArcGISFeatureServerClient:
    def __init__(
        self,
        session: requests.Session | None = None,
        timeout: int = 120,
        force: bool = False,
    ) -> None:
        self.session = session or requests.Session()
        self.timeout = timeout
        self.force = force

    def _get_json(
        self, url: str, params: dict[str, Any], cache_path: Path | None = None
    ) -> dict[str, Any]:
        if cache_path and cache_path.exists() and not self.force:
            with cache_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)

        response = self.session.get(url, params=params, timeout=self.timeout)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            snippet = response.text[:300].replace("\n", " ")
            raise ObservedCountsError(
                f"ArcGIS request failed with HTTP {response.status_code} for {response.url}. "
                f"This is often an upstream FeatureServer outage or changed service URL. "
                f"Response starts with: {snippet}"
            ) from exc
        try:
            data = response.json()
        except ValueError as exc:
            snippet = response.text[:300].replace("\n", " ")
            raise ObservedCountsError(
                f"ArcGIS response was not JSON for {response.url}. Response starts with: {snippet}"
            ) from exc
        if "error" in data:
            raise ObservedCountsError(f"ArcGIS error from {url}: {data['error']}")
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh)
        return data

    def service_metadata(self, service_url: str, cache_dir: Path) -> dict[str, Any]:
        return self._get_json(
            service_url,
            {"f": "json"},
            cache_dir / "service_metadata.json",
        )

    def layer_metadata(self, service_url: str, layer: int, cache_dir: Path) -> dict[str, Any]:
        return self._get_json(
            f"{service_url.rstrip('/')}/{layer}",
            {"f": "json"},
            cache_dir / f"layer_{layer}_metadata.json",
        )

    def download_layer(
        self,
        service_url: str,
        *,
        layer: int = 0,
        cache_dir: Path,
        where: str = "1=1",
        out_fields: str = "*",
        result_record_count: int | None = None,
    ) -> tuple[gpd.GeoDataFrame, dict[str, Any], dict[str, Any]]:
        service_meta = self.service_metadata(service_url, cache_dir)
        layer_meta = self.layer_metadata(service_url, layer, cache_dir)
        max_records = int(result_record_count or layer_meta.get("maxRecordCount") or 2000)

        try:
            gdf = self._download_pages(
                service_url,
                layer,
                cache_dir,
                max_records,
                where,
                out_fields,
                response_format="geojson",
            )
        except Exception as exc:
            LOGGER.warning("GeoJSON download failed for %s layer %s: %s", service_url, layer, exc)
            gdf = self._download_pages(
                service_url,
                layer,
                cache_dir,
                max_records,
                where,
                out_fields,
                response_format="json",
            )

        return gdf, service_meta, layer_meta

    def _download_pages(
        self,
        service_url: str,
        layer: int,
        cache_dir: Path,
        max_records: int,
        where: str,
        out_fields: str,
        *,
        response_format: str,
    ) -> gpd.GeoDataFrame:
        frames: list[gpd.GeoDataFrame] = []
        offset = 0
        page = 0
        query_url = f"{service_url.rstrip('/')}/{layer}/query"
        while True:
            params = {
                "f": response_format,
                "where": where,
                "outFields": out_fields,
                "returnGeometry": "true",
                "resultOffset": offset,
                "resultRecordCount": max_records,
                "outSR": 4326,
            }
            cache_path = cache_dir / f"page_{page:05d}.{response_format}"
            data = self._get_json(query_url, params, cache_path)
            features = data.get("features") or []
            if not features:
                break

            if response_format == "geojson":
                frame = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
            else:
                frame = esri_json_to_geodataframe(data)
            frames.append(frame)

            count = len(features)
            exceeded = bool(data.get("exceededTransferLimit"))
            offset += count
            page += 1
            if count < max_records and not exceeded:
                break

        if not frames:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        out = pd.concat(frames, ignore_index=True)
        return gpd.GeoDataFrame(out, geometry="geometry", crs=frames[0].crs or "EPSG:4326")


def esri_json_to_geodataframe(data: dict[str, Any]) -> gpd.GeoDataFrame:
    rows: list[dict[str, Any]] = []
    geometries = []
    for feature in data.get("features", []):
        attributes = feature.get("attributes", {}) or {}
        geometry = _esri_geometry_to_shapely(feature.get("geometry") or {})
        rows.append(attributes)
        geometries.append(geometry)

    spatial_reference = data.get("spatialReference") or {}
    wkid = spatial_reference.get("latestWkid") or spatial_reference.get("wkid") or 4326
    crs = f"EPSG:{wkid}"
    return gpd.GeoDataFrame(rows, geometry=geometries, crs=crs).to_crs("EPSG:4326")


def _esri_geometry_to_shapely(geometry: dict[str, Any]):
    if "x" in geometry and "y" in geometry:
        return Point(geometry["x"], geometry["y"])
    if "paths" in geometry:
        lines = [LineString(path) for path in geometry["paths"] if len(path) >= 2]
        if not lines:
            return None
        return lines[0] if len(lines) == 1 else MultiLineString(lines)
    if "rings" in geometry:
        polygons = []
        for ring in geometry["rings"]:
            if len(ring) >= 4:
                polygons.append(Polygon(ring))
        if not polygons:
            return None
        return polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)
    return None
