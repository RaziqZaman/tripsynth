#!/usr/bin/env python3
"""Build adjacent-census-tract AADT station screenlines.

This script converts MDOT SHA AADT count locations into tract-pair screenline
candidates by assigning each count point to a Census tract and identifying a
nearby adjacent tract across the closest tract boundary.

Outputs:
  outputs/tract_pair_aadt_station_screenlines.csv
  outputs/tract_pair_screenline_summary_by_group.csv
  outputs/tract_pair_screenline_validation_template.csv
  outputs/tract_pair_screenlines_map.html
  outputs/rejected_or_ambiguous_station_candidates.csv
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import requests
from tqdm import tqdm

try:
    import folium
except Exception:
    folium = None

COUNTY_NAMES = {
    "11001": "District of Columbia",
    "24031": "Montgomery County, MD",
    "24033": "Prince George's County, MD",
}


def read_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"Already have {out_path}")
        return
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def load_tracts(config: dict, root: Path) -> gpd.GeoDataFrame:
    data_dir = root / "data" / "tiger_tracts"
    frames = []
    for state in config["states"]:
        url = config["tract_download_urls"][state]
        zip_path = data_dir / f"tl_{config['census_tiger_year']}_{state}_tract.zip"
        download(url, zip_path)
        shp_name = None
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir / state)
            for name in zf.namelist():
                if name.lower().endswith(".shp"):
                    shp_name = name
        if not shp_name:
            raise RuntimeError(f"No .shp found in {zip_path}")
        shp_path = data_dir / state / shp_name
        gdf = gpd.read_file(shp_path)
        frames.append(gdf)
    tracts = pd.concat(frames, ignore_index=True)
    tracts = gpd.GeoDataFrame(tracts, geometry="geometry", crs=frames[0].crs)
    tracts["TRACT_GEOID"] = tracts["GEOID"].astype(str)
    tracts["COUNTY_GEOID"] = tracts["STATEFP"].astype(str).str.zfill(2) + tracts["COUNTYFP"].astype(str).str.zfill(3)
    tracts["TRACT_LABEL"] = tracts.get("NAMELSAD", tracts["TRACT_GEOID"])
    return tracts


def clean_float(s):
    return pd.to_numeric(s, errors="coerce")


def load_stations(config: dict, root: Path) -> gpd.GeoDataFrame:
    csv_path = root / config["station_input_csv"]
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path, dtype=str)
    df["longitude"] = clean_float(df["longitude"])
    df["latitude"] = clean_float(df["latitude"])
    df = df.dropna(subset=["longitude", "latitude"]).copy()

    # Normalize optional numeric fields.
    for col in ["AADT_2017", "AADT_2018", "AAWDT_2017", "AAWDT_2018", "AADT", "AAWDT", "NUM_LANES", "MAIN_LINE", "ID_RTE_NO", "K_FACTOR", "D_FACTOR"]:
        if col in df.columns:
            df[col + "_NUM"] = clean_float(df[col])

    mask = pd.Series(True, index=df.index)
    if config.get("include_only_seed_candidates", True) and "candidate_for_tract_screenline_seed" in df.columns:
        mask &= df["candidate_for_tract_screenline_seed"].astype(str).str.lower().isin(["true", "1", "yes"])
    if not config.get("include_ramps", False):
        is_ramp = df.get("is_ramp", pd.Series(False, index=df.index)).astype(str).str.lower().isin(["true", "1", "yes"])
        mask &= ~is_ramp
    if not config.get("include_non_mainline", False):
        is_main = df.get("is_mainline", pd.Series(True, index=df.index)).astype(str).str.lower().isin(["true", "1", "yes"])
        mask &= is_main
    df = df[mask].copy()
    df["LOCATION_ID"] = df["LOCATION_ID"].astype(str)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
    return gdf


def make_adjacency(tracts_m: gpd.GeoDataFrame) -> Dict[int, List[int]]:
    """Build touches adjacency among tracts using spatial index."""
    sindex = tracts_m.sindex
    adj = {i: [] for i in range(len(tracts_m))}
    for i, geom in enumerate(tqdm(tracts_m.geometry, desc="Building tract adjacency")):
        cand_idx = list(sindex.query(geom, predicate="touches"))
        adj[i] = [j for j in cand_idx if j != i]
    return adj


def classify_screenline(county_a: str, county_b: str) -> str:
    pair = {county_a, county_b}
    if pair == {"24033", "11001"}:
        return "pg_dc"
    if pair == {"24031", "11001"}:
        return "montgomery_dc"
    if pair == {"24033", "24031"}:
        return "pg_montgomery"
    if "11001" in pair and (county_a.startswith("24") or county_b.startswith("24")):
        return "md_dc_other"
    return "other_adjacent_tract_pair"


def station_pair_quality(boundary_dist_m: float, shared_len_m: float, n_neighbor_candidates: int) -> str:
    if pd.isna(boundary_dist_m):
        return "reject_no_containing_tract"
    if boundary_dist_m <= 50 and shared_len_m >= 50 and n_neighbor_candidates <= 3:
        return "high"
    if boundary_dist_m <= 100 and shared_len_m >= 25:
        return "medium"
    if boundary_dist_m <= 175 and shared_len_m >= 15:
        return "low_review"
    return "reject_far_or_no_shared_boundary"


def assign_pairs(stations: gpd.GeoDataFrame, tracts: gpd.GeoDataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    crs = config.get("analysis_crs", "EPSG:26918")
    max_boundary = float(config.get("max_boundary_distance_m", 175))
    buffer_m = float(config.get("nearby_tract_buffer_m", 225))

    tracts_m = tracts.to_crs(crs).reset_index(drop=True)
    stations_m = stations.to_crs(crs).reset_index(drop=True)
    tracts_m["tract_index"] = tracts_m.index

    # Containing tract join.
    contain = gpd.sjoin(
        stations_m[["LOCATION_ID", "geometry"]],
        tracts_m[["tract_index", "TRACT_GEOID", "COUNTY_GEOID", "TRACT_LABEL", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")
    contain = contain.rename(columns={"TRACT_GEOID":"station_tract_geoid", "COUNTY_GEOID":"station_county_geoid", "TRACT_LABEL":"station_tract_label"})
    station_info = stations_m.drop(columns="geometry").merge(contain.drop(columns="geometry"), on="LOCATION_ID", how="left")

    adj = make_adjacency(tracts_m)
    tract_by_index = tracts_m.set_index("tract_index")
    rows = []
    rejects = []
    for idx, st in tqdm(stations_m.iterrows(), total=len(stations_m), desc="Assigning stations to adjacent tract pairs"):
        loc = st["LOCATION_ID"]
        info = station_info[station_info["LOCATION_ID"] == loc].iloc[0].to_dict()
        tract_idx_val = info.get("tract_index", np.nan)
        if pd.isna(tract_idx_val):
            info.update({"assignment_status":"reject_no_containing_tract"})
            rejects.append(info)
            continue
        tract_idx = int(tract_idx_val)
        point = st.geometry
        tract_geom = tracts_m.loc[tract_idx, "geometry"]
        boundary_dist = float(point.distance(tract_geom.boundary))
        if boundary_dist > max_boundary:
            info.update({"assignment_status":"reject_not_near_tract_boundary", "station_to_boundary_m": boundary_dist})
            rejects.append(info)
            continue
        neighbor_idxs = adj.get(tract_idx, [])
        candidates = []
        for nidx in neighbor_idxs:
            ngeom = tracts_m.loc[nidx, "geometry"]
            # Must be spatially close to the station point or shared edge near the station.
            d_to_neighbor = float(point.distance(ngeom))
            shared = tract_geom.boundary.intersection(ngeom.boundary)
            shared_len = float(shared.length) if not shared.is_empty else 0.0
            d_to_shared = float(point.distance(shared)) if not shared.is_empty else np.inf
            if min(d_to_neighbor, d_to_shared) <= buffer_m:
                candidates.append((nidx, d_to_neighbor, d_to_shared, shared_len))
        if not candidates:
            info.update({"assignment_status":"reject_no_nearby_adjacent_tract", "station_to_boundary_m": boundary_dist})
            rejects.append(info)
            continue
        # Choose the neighbor with nearest shared boundary; tiebreak by shared length descending.
        candidates.sort(key=lambda x: (x[2], -x[3], x[1]))
        nidx, d_to_neighbor, d_to_shared, shared_len = candidates[0]
        nrow = tracts_m.loc[nidx]
        station_county = info.get("station_county_geoid")
        neighbor_county = nrow["COUNTY_GEOID"]
        family = classify_screenline(str(station_county), str(neighbor_county))
        quality = station_pair_quality(float(d_to_shared), float(shared_len), len(candidates))
        tract_a = str(info.get("station_tract_geoid"))
        tract_b = str(nrow["TRACT_GEOID"])
        pair_id = "__".join(sorted([tract_a, tract_b]))
        screenline_id = f"sl_{loc}_{pair_id}"
        out = info.copy()
        out.update({
            "screenline_id": screenline_id,
            "tract_pair_id": pair_id,
            "tract_a_geoid": tract_a,
            "tract_b_geoid": tract_b,
            "station_tract_geoid": tract_a,
            "adjacent_tract_geoid": tract_b,
            "station_county_geoid": station_county,
            "adjacent_county_geoid": neighbor_county,
            "station_county_name": COUNTY_NAMES.get(str(station_county), str(station_county)),
            "adjacent_county_name": COUNTY_NAMES.get(str(neighbor_county), str(neighbor_county)),
            "screenline_family": family,
            "station_to_containing_tract_boundary_m": boundary_dist,
            "station_to_shared_boundary_m": d_to_shared,
            "shared_boundary_length_m": shared_len,
            "n_adjacent_tract_candidates_within_buffer": len(candidates),
            "assignment_quality": quality,
            "assignment_status": "accepted" if not quality.startswith("reject") else quality,
            "validation_observed_daily_field_preferred": "AADT_2018",
            "validation_model_flow_definition": "daily vehicle trips whose OD path crosses this tract-pair boundary; direct adjacent OD only is stricter and will undercount through traffic",
        })
        rows.append(out)
    accepted = pd.DataFrame(rows)
    rejected = pd.DataFrame(rejects)
    return accepted, rejected, tracts_m


def make_validation_template(screenlines: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "screenline_id", "LOCATION_ID", "screenline_family", "tract_pair_id", "tract_a_geoid", "tract_b_geoid",
        "ROADNAME", "ID_PREFIX", "ID_RTE_NO", "STATION_DESC", "PEAK_HOUR_DIRECTION", "NUM_LANES",
        "AADT_2017", "AADT_2018", "AAWDT_2017", "AAWDT_2018", "COUNTED_FACTORED", "assignment_quality",
    ]
    out = screenlines[[c for c in cols if c in screenlines.columns]].copy()
    out["model_daily_vehicle_flow_2017"] = ""
    out["model_daily_vehicle_flow_2018"] = ""
    out["abs_error_2018"] = "=model_daily_vehicle_flow_2018-AADT_2018"
    out["pct_error_2018"] = "=(model_daily_vehicle_flow_2018-AADT_2018)/AADT_2018"
    out["validation_notes"] = ""
    return out


def make_map(screenlines: pd.DataFrame, out_html: Path) -> None:
    if folium is None or screenlines.empty:
        return
    center = [screenlines["latitude"].astype(float).mean(), screenlines["longitude"].astype(float).mean()]
    m = folium.Map(location=center, zoom_start=10, tiles="CartoDB positron")
    colors = {
        "pg_dc": "red",
        "montgomery_dc": "green",
        "pg_montgomery": "blue",
        "md_dc_other": "purple",
        "other_adjacent_tract_pair": "gray",
    }
    for _, r in screenlines.iterrows():
        lat = pd.to_numeric(r.get("latitude"), errors="coerce")
        lon = pd.to_numeric(r.get("longitude"), errors="coerce")
        if pd.isna(lat) or pd.isna(lon):
            continue
        fam = r.get("screenline_family", "other_adjacent_tract_pair")
        popup = "<br>".join([
            f"<b>{r.get('LOCATION_ID','')}</b>",
            f"family: {fam}",
            f"tract pair: {r.get('tract_pair_id','')}",
            f"road: {r.get('ROADNAME','')} / {r.get('ID_PREFIX','')} {r.get('ID_RTE_NO','')}",
            f"desc: {r.get('STATION_DESC','')}",
            f"AADT 2017: {r.get('AADT_2017','')}",
            f"AADT 2018: {r.get('AADT_2018','')}",
            f"quality: {r.get('assignment_quality','')}",
            f"dist to shared boundary m: {r.get('station_to_shared_boundary_m','')}",
            f"<a href='{r.get('LINK','')}' target='_blank'>I-TMS link</a>" if str(r.get('LINK','')).startswith('http') else "",
        ])
        folium.CircleMarker(
            location=[lat, lon], radius=5,
            color=colors.get(fam, "gray"), fill=True, fill_opacity=0.75,
            popup=folium.Popup(popup, max_width=450),
            tooltip=f"{r.get('LOCATION_ID','')} | {fam} | {r.get('assignment_quality','')}"
        ).add_to(m)
    folium.LayerControl().add_to(m)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    root = Path.cwd()
    config = read_config(root / args.config)
    out_dir = root / "outputs"
    out_dir.mkdir(exist_ok=True, parents=True)

    print("Loading MDOT AADT station candidates...")
    stations = load_stations(config, root)
    print(f"Station candidates after filters: {len(stations):,}")

    print("Loading Census TIGER tracts...")
    tracts = load_tracts(config, root)
    print(f"Tracts loaded: {len(tracts):,}")

    screenlines, rejected, tracts_m = assign_pairs(stations, tracts, config)

    # Keep a sensible column order.
    first_cols = [
        "screenline_id", "tract_pair_id", "screenline_family", "assignment_quality", "assignment_status",
        "LOCATION_ID", "COUNTY_DESC", "ROADNAME", "ID_PREFIX", "ID_RTE_NO", "ID_MP", "STATION_DESC", "ROAD_SECTION",
        "PEAK_HOUR_DIRECTION", "NUM_LANES", "COUNTED_FACTORED", "AADT_2017", "AADT_2018", "AAWDT_2017", "AAWDT_2018",
        "K_FACTOR", "D_FACTOR", "NORTH_EAST_SPLIT", "SOUTH_WEST_SPLIT",
        "station_tract_geoid", "adjacent_tract_geoid", "station_county_geoid", "adjacent_county_geoid",
        "station_to_shared_boundary_m", "shared_boundary_length_m", "n_adjacent_tract_candidates_within_buffer",
        "longitude", "latitude", "LINK"
    ]
    cols = [c for c in first_cols if c in screenlines.columns] + [c for c in screenlines.columns if c not in first_cols]
    screenlines = screenlines[cols] if not screenlines.empty else screenlines
    screenlines.to_csv(out_dir / "tract_pair_aadt_station_screenlines.csv", index=False)
    rejected.to_csv(out_dir / "rejected_or_ambiguous_station_candidates.csv", index=False)

    if not screenlines.empty:
        summary = (screenlines.groupby(["screenline_family", "assignment_quality"], dropna=False)
                   .size().reset_index(name="n_station_screenlines"))
    else:
        summary = pd.DataFrame(columns=["screenline_family", "assignment_quality", "n_station_screenlines"])
    summary.to_csv(out_dir / "tract_pair_screenline_summary_by_group.csv", index=False)

    template = make_validation_template(screenlines)
    template.to_csv(out_dir / "tract_pair_screenline_validation_template.csv", index=False)
    make_map(screenlines, out_dir / "tract_pair_screenlines_map.html")

    report = []
    report.append("# Adjacent Census Tract AADT Station Screenline Report\n")
    report.append(f"Station candidates after filters: {len(stations):,}\n")
    report.append(f"Accepted tract-pair screenlines: {len(screenlines):,}\n")
    report.append(f"Rejected/ambiguous candidates: {len(rejected):,}\n")
    report.append("\n## Screenline summary\n")
    report.append(summary.to_markdown(index=False) if not summary.empty else "No accepted screenlines.")
    report.append("\n\n## Method\n")
    report.append(
        "Each MDOT AADT count location is assigned to its containing Census tract. "
        "A station is accepted as a tract-pair screenline only if it lies within the configured distance "
        "of the containing tract boundary and has a nearby adjacent tract sharing that boundary. "
        "This makes the output a station-level tract-boundary screenline inventory, not an OD matrix."
    )
    (out_dir / "RUN_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("\nDone. Start with outputs/RUN_REPORT.md")


if __name__ == "__main__":
    main()
