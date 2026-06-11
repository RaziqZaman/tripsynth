from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import requests

from trip_synth.utils.io import ensure_dir, load_yaml, write_json

from .arcgis import discover_line_layer, query_feature_layer, request_json


def _download_file(url: str, output: Path, session: requests.Session) -> None:
    ensure_dir(output.parent)
    with session.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with output.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _validate_geofile(path: Path) -> dict[str, Any]:
    try:
        import geopandas as gpd  # type: ignore

        gdf = gpd.read_file(path)
        return {"path": str(path), "status": "ok", "rows": int(len(gdf))}
    except Exception as exc:
        return {"path": str(path), "status": "not_validated", "reason": str(exc)}


def download_external_data(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    base = Path(config.get("external_data_dir", "data/external"))
    ensure_dir(base)
    session = requests.Session()
    manifest: dict[str, Any] = {
        "config": str(config_path),
        "files": [],
        "warnings": [],
        "attempted_sources": [],
    }

    tiger = config.get("census_tiger", {})
    for name, state in (tiger.get("states", {}) or {}).items():
        url = state["tract_zip_url"]
        out_dir = Path(state["output_dir"])
        zip_path = out_dir / Path(url).name
        manifest["attempted_sources"].append(url)
        try:
            if not zip_path.exists():
                _download_file(url, zip_path, session)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(out_dir)
            shp = out_dir / f"tl_{tiger.get('year', 2020)}_{state['state_fips']}_tract.shp"
            manifest["files"].append(_validate_geofile(shp))
        except Exception as exc:
            manifest["warnings"].append(
                {
                    "source": "census_tiger",
                    "state": name,
                    "url": url,
                    "error": str(exc),
                    "instruction": "Check network access and rerun scripts/download_external_data.py.",
                }
            )

    mdot = config.get("mdot_aadt", {})
    if mdot:
        service_root = mdot["service_root"]
        out_dir = ensure_dir(mdot.get("output_dir", "data/external/mdot_aadt"))
        manifest["attempted_sources"].append(service_root)
        try:
            metadata = request_json(service_root, {"f": "json"}, session=session)
            write_json(metadata, out_dir / "aadt_service_metadata.json")
            points_geojson, points_meta = query_feature_layer(
                service_root,
                int(mdot.get("points_layer_id", 0)),
                where=str(mdot.get("where", "1=1")),
                out_sr=int(mdot.get("out_sr", 4326)),
                page_size=int(mdot.get("page_size", 20000)),
                prefer_geojson=str(mdot.get("query_format", "geojson")).lower() == "geojson",
                session=session,
            )
            (out_dir / "aadt_points.geojson").write_text(json.dumps(points_geojson))
            manifest["files"].append(_validate_geofile(out_dir / "aadt_points.geojson"))
            line_id = discover_line_layer(
                metadata,
                [int(v) for v in mdot.get("candidate_lines_layer_ids", [])],
            )
            if line_id is not None:
                segments_geojson, _ = query_feature_layer(
                    service_root,
                    line_id,
                    where=str(mdot.get("where", "1=1")),
                    out_sr=int(mdot.get("out_sr", 4326)),
                    page_size=int(mdot.get("page_size", 20000)),
                    prefer_geojson=str(mdot.get("query_format", "geojson")).lower() == "geojson",
                    session=session,
                )
                (out_dir / "aadt_segments.geojson").write_text(json.dumps(segments_geojson))
                manifest["files"].append(_validate_geofile(out_dir / "aadt_segments.geojson"))
            else:
                manifest["warnings"].append(
                    {
                        "source": "mdot_aadt",
                        "error": "No line/segment layer discovered",
                        "instruction": "Inspect aadt_service_metadata.json and update candidate_lines_layer_ids.",
                    }
                )
            manifest["mdot_points_metadata"] = {
                "maxRecordCount": points_meta.get("maxRecordCount"),
                "fields": [f.get("name") for f in points_meta.get("fields", [])],
            }
        except Exception as exc:
            manifest["warnings"].append(
                {
                    "source": "mdot_aadt",
                    "url": service_root,
                    "error": str(exc),
                    "instruction": "Confirm the MDOT ArcGIS service is reachable and rerun the downloader.",
                }
            )

    write_json(manifest, base / "DATA_MANIFEST.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_sources.yaml")
    args = parser.parse_args()
    manifest = download_external_data(args.config)
    print(json.dumps(manifest, indent=2, default=str))
