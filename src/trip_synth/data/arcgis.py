from __future__ import annotations

import time
from typing import Any

import requests


LINE_LAYER_KEYWORDS = ("line", "lines", "segment", "segments")


def request_json(
    url: str,
    params: dict[str, Any] | None = None,
    retries: int = 3,
    timeout: int = 60,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    session = session or requests.Session()
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed ArcGIS request after {retries} attempts: {url}") from last_error


def discover_line_layer(metadata: dict[str, Any], candidate_ids: list[int] | None = None) -> int | None:
    layers = metadata.get("layers", []) or []
    by_id = {int(layer.get("id")): layer for layer in layers if layer.get("id") is not None}
    for layer_id in candidate_ids or []:
        if int(layer_id) in by_id:
            return int(layer_id)
    for layer in layers:
        text = " ".join(
            str(layer.get(k, "")) for k in ["name", "description", "type", "displayField"]
        ).lower()
        if any(word in text for word in LINE_LAYER_KEYWORDS):
            return int(layer["id"])
    return None


def esri_json_to_geojson(esri: dict[str, Any]) -> dict[str, Any]:
    features = []
    geometry_type = str(esri.get("geometryType", "")).lower()
    for feature in esri.get("features", []):
        attrs = feature.get("attributes", {})
        geom = feature.get("geometry", {})
        geometry = None
        if "x" in geom and "y" in geom:
            geometry = {"type": "Point", "coordinates": [geom["x"], geom["y"]]}
        elif "paths" in geom:
            coords = geom.get("paths", [])
            if len(coords) == 1:
                geometry = {"type": "LineString", "coordinates": coords[0]}
            elif coords:
                geometry = {"type": "MultiLineString", "coordinates": coords}
        elif "rings" in geom:
            geometry = {"type": "Polygon", "coordinates": geom.get("rings", [])}
        if geometry is None and "point" in geometry_type:
            geometry = None
        features.append({"type": "Feature", "properties": attrs, "geometry": geometry})
    return {"type": "FeatureCollection", "features": features}


def query_feature_layer(
    service_root: str,
    layer_id: int,
    where: str = "1=1",
    out_sr: int = 4326,
    page_size: int = 20000,
    prefer_geojson: bool = True,
    session: requests.Session | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    session = session or requests.Session()
    layer_url = f"{service_root.rstrip('/')}/{layer_id}"
    metadata = request_json(layer_url, {"f": "json"}, session=session)
    max_record_count = int(metadata.get("maxRecordCount", page_size) or page_size)
    take = min(int(page_size), max_record_count)
    offset = 0
    all_features: list[dict[str, Any]] = []
    used_format = "geojson" if prefer_geojson else "json"
    while True:
        fmt = "geojson" if used_format == "geojson" else "json"
        params = {
            "f": fmt,
            "where": where,
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": out_sr,
            "resultOffset": offset,
            "resultRecordCount": take,
        }
        try:
            payload = request_json(f"{layer_url}/query", params, session=session)
            if fmt == "json" or "features" not in payload or payload.get("type") != "FeatureCollection":
                payload = esri_json_to_geojson(payload)
        except RuntimeError:
            if used_format == "geojson":
                used_format = "json"
                continue
            raise
        features = payload.get("features", [])
        all_features.extend(features)
        exceeded = bool(payload.get("exceededTransferLimit"))
        if len(features) < take and not exceeded:
            break
        offset += take
    return {"type": "FeatureCollection", "features": all_features}, metadata
