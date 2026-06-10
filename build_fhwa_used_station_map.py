#!/usr/bin/env python3
"""
Build an interactive Leaflet/Folium map of every station-direction combination
that appears in an FHWA TMAS hourly-count extract.

Inputs
------
1. Hourly count CSV, with at least:
   state_code, station_id, direction, lane, date, hour, volume

2. FHWA station metadata CSV/TXT, with at least:
   state_code, station_id, direction, latitude, longitude, route, location_text

The station metadata can be either:
- a clean delimited table such as Station_Data_Extract_Pipe_Delimited_CleanData_2018.txt, or
- a CSV you exported after parsing FHWA 2018 station data.

If station metadata is not supplied, this script attempts to download the official
FHWA 2018 station-data ZIP and parse delimited tables inside it. If the official
file is fixed-width on your machine, pass a cleaned station metadata CSV/TXT with
--station-metadata.

Official FHWA source URLs
-------------------------
2018 Station Data:
https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/2018/2018_station_data.zip
June 2018 CCS hourly volume data:
https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/2018/jun_2018_ccs_data.zip

Example
-------
python build_fhwa_used_station_map.py \
  --hourly /path/to/fhwa_2018_06_10_all_station_hourly.csv \
  --station-metadata /path/to/Station_Data_Extract_Pipe_Delimited_CleanData_2018.txt \
  --outdir outputs
"""
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

STATION_ZIP_URL = "https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/2018/2018_station_data.zip"

DIRECTION_LABELS = {"1": "northbound", "3": "eastbound", "5": "southbound", "7": "westbound"}


def norm_state(x):
    s = "" if pd.isna(x) else str(x).strip()
    s = re.sub(r"\D", "", s)
    return s.zfill(2)[-2:] if s else ""


def norm_station(x):
    return "" if pd.isna(x) else str(x).strip().upper()


def norm_dir(x):
    s = "" if pd.isna(x) else str(x).strip().upper()
    return re.sub(r"\.0$", "", s)


def norm_lane(x):
    s = "" if pd.isna(x) else str(x).strip().upper()
    return re.sub(r"\.0$", "", s)


def detect_col(cols, candidates):
    """Find a likely column from a set of candidate names."""
    norm = {re.sub(r"[^A-Z0-9]", "", c.upper()): c for c in cols}
    for cand in candidates:
        key = re.sub(r"[^A-Z0-9]", "", cand.upper())
        if key in norm:
            return norm[key]
    for c in cols:
        cc = re.sub(r"[^A-Z0-9]", "", c.upper())
        for cand in candidates:
            key = re.sub(r"[^A-Z0-9]", "", cand.upper())
            if key and key in cc:
                return c
    return None


def read_delimited(path: Path) -> pd.DataFrame:
    # Try sniffing pipe/comma/tab. Python engine tolerates ragged long text better.
    sample = path.read_text(encoding="latin1", errors="ignore")[:10000]
    if "|" in sample:
        sep = "|"
    elif "\t" in sample:
        sep = "\t"
    else:
        sep = ","
    return pd.read_csv(path, sep=sep, dtype=str, low_memory=False, engine="python", on_bad_lines="skip")


def read_metadata_file(path: Path) -> pd.DataFrame:
    df = read_delimited(path)
    cols = list(df.columns)
    st_col = detect_col(cols, ["state_code", "state", "fips_state_code", "fips_state"])
    sta_col = detect_col(cols, ["station_id", "station", "stationid", "station_number"])
    dir_col = detect_col(cols, ["direction", "dir", "direction_of_travel"])
    lat_col = detect_col(cols, ["latitude", "lat"])
    lon_col = detect_col(cols, ["longitude", "lon", "long"])
    route_col = detect_col(cols, ["route", "route_id", "route_name"])
    loc_col = detect_col(cols, ["location_text", "location", "station_location", "description"])
    src_col = detect_col(cols, ["source_file"])

    if not (st_col and sta_col and lat_col and lon_col):
        raise ValueError(
            f"Could not identify required station metadata columns in {path}. "
            f"Detected columns: {cols[:20]}"
        )

    out = pd.DataFrame({
        "state_code": df[st_col].map(norm_state),
        "station_id": df[sta_col].map(norm_station),
        "direction": df[dir_col].map(norm_dir) if dir_col else "",
        "latitude": pd.to_numeric(df[lat_col], errors="coerce"),
        "longitude": pd.to_numeric(df[lon_col], errors="coerce"),
        "route": df[route_col].astype(str) if route_col else "",
        "location_text": df[loc_col].astype(str) if loc_col else "",
        "source_file": df[src_col].astype(str) if src_col else path.name,
    })
    return clean_coords(out)


def clean_coords(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # FHWA-style coordinates in the user's extract look like 38918060,76944910.
    for col in ["latitude", "longitude"]:
        big = df[col].abs() > 180
        df.loc[big, col] = df.loc[big, col] / 1_000_000.0
    # Most longitude values in MD/DC/VA station extract are positive west-longitude magnitudes.
    df.loc[df["longitude"] > 0, "longitude"] = -df.loc[df["longitude"] > 0, "longitude"]
    # Keep only plausible Mid-Atlantic-ish coordinates; zeros become missing.
    df.loc[~df["latitude"].between(35, 41), "latitude"] = pd.NA
    df.loc[~df["longitude"].between(-82, -74), "longitude"] = pd.NA
    return df


def download_station_zip(outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / "2018_station_data.zip"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    req = Request(STATION_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=180) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    return dest


def read_metadata_from_zip(zip_path: Path, outdir: Path) -> pd.DataFrame:
    extract_dir = outdir / "station_zip_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    frames = []
    for p in extract_dir.rglob("*"):
        if not p.is_file():
            continue
        try:
            df = read_metadata_file(p)
            if {"state_code", "station_id", "latitude", "longitude"}.issubset(df.columns):
                frames.append(df)
        except Exception:
            continue
    if not frames:
        raise RuntimeError(
            "Downloaded station ZIP, but no clean delimited metadata table could be parsed. "
            "Use FHWA's conversion tool or pass --station-metadata with a clean CSV/TXT."
        )
    return pd.concat(frames, ignore_index=True)


def summarize_hourly(hourly_path: Path) -> pd.DataFrame:
    h = pd.read_csv(hourly_path, dtype=str, low_memory=False)
    required = {"state_code", "station_id", "direction", "lane", "hour", "volume"}
    missing = required - set(h.columns)
    if missing:
        raise ValueError(f"Hourly file is missing required columns: {sorted(missing)}")
    h["state_code"] = h["state_code"].map(norm_state)
    h["station_id"] = h["station_id"].map(norm_station)
    h["direction"] = h["direction"].map(norm_dir)
    h["lane"] = h["lane"].map(norm_lane)
    h["volume"] = pd.to_numeric(h["volume"], errors="coerce").fillna(0)
    # Preserve all lanes and hours, but map station-direction once.
    g = (
        h.groupby(["state_code", "station_id", "direction"], as_index=False)
        .agg(
            lanes=("lane", lambda x: ",".join(sorted(set(map(str, x))))),
            n_lane_hour_records=("volume", "size"),
            total_volume=("volume", "sum"),
            max_hourly_lane_volume=("volume", "max"),
        )
    )
    g["direction_label"] = g["direction"].map(DIRECTION_LABELS).fillna("")
    return g


def join_used_to_metadata(used: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    meta["state_code"] = meta["state_code"].map(norm_state)
    meta["station_id"] = meta["station_id"].map(norm_station)
    meta["direction"] = meta["direction"].map(norm_dir)
    # Use direction-level metadata when present; otherwise station-level fallback.
    meta_dir = meta.dropna(subset=["latitude", "longitude"]).drop_duplicates(["state_code", "station_id", "direction"])
    j = used.merge(meta_dir, on=["state_code", "station_id", "direction"], how="left", suffixes=("", "_meta"))
    no_coord = j["latitude"].isna() | j["longitude"].isna()
    if no_coord.any():
        meta_station = (
            meta.dropna(subset=["latitude", "longitude"])
            .sort_values(["state_code", "station_id", "direction"])
            .drop_duplicates(["state_code", "station_id"])
            [["state_code", "station_id", "latitude", "longitude", "route", "location_text", "source_file"]]
        )
        fallback = used.loc[no_coord, ["state_code", "station_id", "direction"]].merge(meta_station, on=["state_code", "station_id"], how="left")
        for col in ["latitude", "longitude", "route", "location_text", "source_file"]:
            j.loc[no_coord, col] = fallback[col].values
    j["has_coordinates"] = j["latitude"].notna() & j["longitude"].notna()
    return j


def write_folium_map(joined: pd.DataFrame, html_path: Path) -> None:
    import folium
    from folium.plugins import MarkerCluster, FeatureGroupSubGroup, Fullscreen, MeasureControl

    mapped = joined[joined["has_coordinates"]].copy()
    if mapped.empty:
        raise RuntimeError("No station coordinates after join; cannot create map.")
    center = [mapped["latitude"].mean(), mapped["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=8, tiles="CartoDB positron", control_scale=True)
    Fullscreen().add_to(m)
    MeasureControl(position="topleft", primary_length_unit="miles").add_to(m)

    colors = {"11": "red", "24": "blue", "51": "green"}
    state_names = {"11": "DC", "24": "Maryland", "51": "Virginia"}

    for st in sorted(mapped["state_code"].dropna().unique()):
        fg = folium.FeatureGroup(name=f"{state_names.get(st, st)} used stations", show=True)
        cluster = MarkerCluster(name=f"{state_names.get(st, st)} cluster")
        fg.add_child(cluster)
        for _, r in mapped[mapped["state_code"] == st].iterrows():
            popup = f"""
            <b>{r['state_code']} / {r['station_id']} / dir {r['direction']}</b><br>
            Direction: {r.get('direction_label','')}<br>
            Route: {r.get('route','')}<br>
            Location: {r.get('location_text','')}<br>
            Lanes in hourly file: {r.get('lanes','')}<br>
            June 10 total lane-volume sum: {int(r.get('total_volume',0)):,}<br>
            Records: {int(r.get('n_lane_hour_records',0)):,}<br>
            Source: {r.get('source_file','')}
            """
            folium.CircleMarker(
                location=[r["latitude"], r["longitude"]],
                radius=5,
                color=colors.get(st, "gray"),
                fill=True,
                fill_opacity=0.75,
                weight=1,
                tooltip=f"{r['state_code']} {r['station_id']} dir {r['direction']} | {r.get('location_text','')}",
                popup=folium.Popup(popup, max_width=450),
            ).add_to(cluster)
        fg.add_to(m)

    # Approximate review zones for intended screenlines. These are NOT official screenlines.
    zones = {
        "PG-DC review zone": [[38.78, -77.05], [39.02, -76.83]],
        "Montgomery-DC review zone": [[38.92, -77.18], [39.06, -76.95]],
        "PG-Montgomery review zone": [[38.96, -77.08], [39.12, -76.88]],
        "DC-Virginia / Potomac review zone": [[38.78, -77.18], [39.00, -77.02]],
    }
    zone_fg = folium.FeatureGroup(name="Approximate screenline review zones", show=True)
    for name, bounds in zones.items():
        folium.Rectangle(bounds=bounds, tooltip=name, popup=name, color="black", weight=1, fill=False, dash_array="5,5").add_to(zone_fg)
    zone_fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(html_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hourly", default="/mnt/data/Pasted text(20).txt", help="Hourly count CSV")
    ap.add_argument("--station-metadata", default=None, help="Clean FHWA station metadata CSV/TXT")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    hourly = Path(args.hourly)
    used = summarize_hourly(hourly)
    used_path = outdir / "fhwa_hourly_used_station_directions.csv"
    used.to_csv(used_path, index=False)

    if args.station_metadata:
        meta = read_metadata_file(Path(args.station_metadata))
    else:
        zip_path = download_station_zip(outdir)
        meta = read_metadata_from_zip(zip_path, outdir)
    meta_path = outdir / "fhwa_station_metadata_parsed_for_map.csv"
    meta.to_csv(meta_path, index=False)

    joined = join_used_to_metadata(used, meta)
    joined_path = outdir / "fhwa_used_stations_joined_to_metadata.csv"
    joined.to_csv(joined_path, index=False)

    missing_path = outdir / "fhwa_used_stations_missing_coordinates.csv"
    joined.loc[~joined["has_coordinates"]].to_csv(missing_path, index=False)

    html_path = outdir / "fhwa_used_station_map.html"
    write_folium_map(joined, html_path)

    print("Wrote:")
    for p in [used_path, meta_path, joined_path, missing_path, html_path]:
        print(f"  {p}")
    print("\nCoverage summary:")
    print(joined.groupby("state_code").agg(
        station_directions=("station_id", "size"),
        mapped=("has_coordinates", "sum"),
        stations=("station_id", "nunique")
    ))


if __name__ == "__main__":
    main()
