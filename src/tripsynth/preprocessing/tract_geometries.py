"""Census tract geometry preparation for tract-level routing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests

from tripsynth.config import ensure_standard_directories, flatten_study_counties, resolve_path, write_json


STATE_ABBREVIATION_TO_FIPS = {"dc": "11", "md": "24", "va": "51"}


@dataclass(frozen=True)
class TractGeometryResult:
    tracts: gpd.GeoDataFrame
    manifest: dict[str, Any]
    output_paths: dict[str, str]
    used_cache: bool = False


def normalize_tract_fips(value: Any) -> str | None:
    """Normalize an 11-digit Census tract GEOID from numeric or text survey values."""
    if pd.isna(value):
        return None
    text = str(value).strip().replace(".0", "")
    if not text or text.lower() == "nan":
        return None
    try:
        text = str(int(float(text)))
    except ValueError:
        text = "".join(ch for ch in text if ch.isdigit())
    if len(text) < 11:
        return None
    return text.zfill(11)[:11]


def _tract_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("census", {}).get("tracts", {})


def _processed_path(config: dict[str, Any]) -> Path:
    value = _tract_config(config).get(
        "processed_path", "data/processed/census/tiger_tracts_study_area.geoparquet"
    )
    path = resolve_path(config, value)
    if path is None:
        raise ValueError("census.tracts.processed_path could not be resolved.")
    return path


def _raw_dir(config: dict[str, Any]) -> Path:
    value = _tract_config(config).get("raw_dir", "data/raw/census/tiger_tracts")
    path = resolve_path(config, value)
    if path is None:
        raise ValueError("census.tracts.raw_dir could not be resolved.")
    return path


def _configured_state_fips(config: dict[str, Any]) -> list[str]:
    configured = _tract_config(config).get("states")
    if configured:
        states = []
        for value in configured:
            text = str(value).strip().lower()
            states.append(STATE_ABBREVIATION_TO_FIPS.get(text, text.zfill(2)))
        return sorted(set(states))
    return sorted({county[:2] for county in flatten_study_counties(config) if len(county) >= 2})


def _tiger_tract_url(year: int, state_fips: str) -> str:
    return (
        f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/"
        f"tl_{year}_{state_fips}_tract.zip"
    )


def _download_zip(url: str, path: Path, *, force: bool) -> bool:
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    path.write_bytes(response.content)
    return True


def _read_tiger_zip(path: Path) -> gpd.GeoDataFrame:
    return gpd.read_file(f"zip://{path.resolve()}")


def _standardize_tracts(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    columns = {column.lower(): column for column in gdf.columns}
    geoid = columns.get("geoid")
    statefp = columns.get("statefp")
    countyfp = columns.get("countyfp")
    tractce = columns.get("tractce")
    name = columns.get("name")
    if not geoid:
        raise ValueError("TIGER tract file does not contain a GEOID column.")

    out = gdf.copy()
    out["tract_fips"] = out[geoid].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(11)
    if statefp and countyfp:
        out["state_fips"] = out[statefp].astype("string").str.zfill(2)
        out["county_fips"] = (
            out[statefp].astype("string").str.zfill(2) + out[countyfp].astype("string").str.zfill(3)
        )
    else:
        out["state_fips"] = out["tract_fips"].str[:2]
        out["county_fips"] = out["tract_fips"].str[:5]
    out["tract_code"] = out[tractce].astype("string").str.zfill(6) if tractce else out["tract_fips"].str[5:]
    out["tract_name"] = out[name].astype("string") if name else pd.NA
    out = out[["tract_fips", "state_fips", "county_fips", "tract_code", "tract_name", "geometry"]]
    return gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs)


def fetch_tiger_tracts(config: dict[str, Any], *, force: bool = False) -> TractGeometryResult:
    """Download, normalize, and cache Census TIGER/Line tract polygons."""
    ensure_standard_directories()
    tracts_config = _tract_config(config)
    year = int(tracts_config.get("year", 2018))
    state_fips = _configured_state_fips(config)
    if not state_fips:
        raise ValueError("No states found for Census tract fetch. Configure census.tracts.states.")

    processed_path = _processed_path(config)
    manifest_path = Path("data/metadata/census_tract_geometries_manifest.json")
    if processed_path.exists() and manifest_path.exists() and not force:
        return TractGeometryResult(
            tracts=gpd.read_parquet(processed_path),
            manifest={},
            output_paths={"geoparquet": str(processed_path), "manifest_json": str(manifest_path)},
            used_cache=True,
        )

    raw_dir = _raw_dir(config)
    downloads = []
    frames = []
    for state in state_fips:
        url = _tiger_tract_url(year, state)
        zip_path = raw_dir / f"tl_{year}_{state}_tract.zip"
        downloaded = _download_zip(url, zip_path, force=force)
        frames.append(_standardize_tracts(_read_tiger_zip(zip_path)))
        downloads.append({"state_fips": state, "url": url, "path": str(zip_path), "downloaded": downloaded})

    tracts = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)
    counties = flatten_study_counties(config)
    if counties:
        tracts = tracts.loc[tracts["county_fips"].astype("string").isin(counties)].copy()

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    tracts.to_parquet(processed_path, index=False)

    manifest = {
        "source_name": "census_tiger_tracts",
        "year": year,
        "download_timestamp": datetime.now(timezone.utc).isoformat(),
        "states": state_fips,
        "downloads": downloads,
        "processed_path": str(processed_path),
        "records": int(len(tracts)),
        "records_by_county": tracts.groupby("county_fips").size().astype(int).to_dict() if len(tracts) else {},
        "crs": str(tracts.crs),
        "bounds": list(map(float, tracts.total_bounds)) if len(tracts) else None,
    }
    write_json(manifest_path, manifest)
    return TractGeometryResult(
        tracts=tracts,
        manifest=manifest,
        output_paths={"geoparquet": str(processed_path), "manifest_json": str(manifest_path)},
        used_cache=False,
    )


def load_or_fetch_tract_geometries(
    config: dict[str, Any], *, allow_fetch: bool = False, force: bool = False
) -> TractGeometryResult:
    processed_path = _processed_path(config)
    manifest_path = Path("data/metadata/census_tract_geometries_manifest.json")
    if processed_path.exists() and not force:
        return TractGeometryResult(
            tracts=gpd.read_parquet(processed_path),
            manifest={},
            output_paths={"geoparquet": str(processed_path), "manifest_json": str(manifest_path)},
            used_cache=True,
        )
    if not allow_fetch:
        raise FileNotFoundError(
            f"Tract geometries are not cached at {processed_path}. "
            "Run the tract fetch step or set validation.routing.mdot_shortest_path.auto_fetch_tracts=true."
        )
    return fetch_tiger_tracts(config, force=force)


def load_or_fetch_tract_centroids(
    config: dict[str, Any],
    projected_crs: str,
    *,
    allow_fetch: bool = False,
    force: bool = False,
) -> gpd.GeoDataFrame:
    result = load_or_fetch_tract_geometries(config, allow_fetch=allow_fetch, force=force)
    tracts = result.tracts
    if tracts.empty:
        return gpd.GeoDataFrame(
            columns=["tract_fips", "county_fips", "geometry"], geometry="geometry", crs=projected_crs
        )
    projected = tracts.to_crs(projected_crs).copy()
    projected["geometry"] = projected.geometry.centroid
    return gpd.GeoDataFrame(
        projected[["tract_fips", "county_fips", "geometry"]],
        geometry="geometry",
        crs=projected_crs,
    )


def tract_geometry_requirements() -> str:
    return (
        "Provide or fetch Census TIGER/Line tract polygons. The default source is "
        "https://www2.census.gov/geo/tiger/TIGER2018/TRACT/."
    )
