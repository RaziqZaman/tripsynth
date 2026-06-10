#!/usr/bin/env python3
"""Build an interactive map of FHWA stations for DC area screenline review.

The map is meant for selecting count stations for three screenline families:
DC-Prince George's County, Prince George's-Montgomery County, and
Montgomery County-DC. It joins the June 10, 2018 hourly TMAS counts to FHWA
station metadata, keeps Maryland/DC stations near those boundaries, and writes
both a Leaflet HTML map and candidate station-map CSVs.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_HOURLY = Path("outputs/fhwa_2018_06_10_all_station_hourly.csv")
DEFAULT_METADATA_PARSED = Path("outputs/fhwa_2018_06_10_station_metadata_parsed.csv")
DEFAULT_METADATA_RAW = Path("fhwa_tmas_raw/station_2018/Station_Data_Extract_Pipe_Delimited_CleanData_2018.txt")
DEFAULT_OUTDIR = Path("outputs")

STATE_NAMES = {"11": "District of Columbia", "24": "Maryland", "51": "Virginia"}
MD_COUNTIES = {"031": "Montgomery County", "033": "Prince George's County"}
DIRECTION_LABELS = {"1": "northbound", "3": "eastbound", "5": "southbound", "7": "westbound"}

# Broad review areas, not official screenline geometry.
SCREENLINE_DEFINITIONS = {
    "dc_pg": {
        "label": "DC-Prince George's County",
        "color": "#d73027",
        "bounds": [[38.79, -77.06], [39.03, -76.82]],
        "state_codes": {"11", "24"},
        "county_codes": {"001", "033"},
    },
    "pg_montgomery": {
        "label": "Prince George's-Montgomery County",
        "color": "#4575b4",
        "bounds": [[38.91, -77.12], [39.22, -76.82]],
        "state_codes": {"24"},
        "county_codes": {"031", "033"},
    },
    "montgomery_dc": {
        "label": "Montgomery County-DC",
        "color": "#1a9850",
        "bounds": [[38.88, -77.22], [39.12, -76.94]],
        "state_codes": {"11", "24"},
        "county_codes": {"001", "031"},
    },
}

MAP_COLUMNS = ["include", "screenline", "state_code", "station_id", "direction", "lane", "station_label", "notes"]


def normalize_state(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    text = re.sub(r"\D", "", text)
    return text.zfill(2)[-2:] if text else ""


def normalize_county(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    text = re.sub(r"\D", "", text)
    return text.zfill(3)[-3:] if text else ""


def normalize_station(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip().upper()


def normalize_code(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().upper()
    return re.sub(r"\.0$", "", text)


def detect_col(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {re.sub(r"[^A-Z0-9]", "", col.upper()): col for col in columns}
    for candidate in candidates:
        key = re.sub(r"[^A-Z0-9]", "", candidate.upper())
        if key in normalized:
            return normalized[key]
    for col in columns:
        compact = re.sub(r"[^A-Z0-9]", "", col.upper())
        for candidate in candidates:
            key = re.sub(r"[^A-Z0-9]", "", candidate.upper())
            if key and key in compact:
                return col
    return None


def read_delimited(path: Path) -> pd.DataFrame:
    sample = path.read_text(encoding="latin1", errors="ignore")[:10000]
    if "|" in sample:
        sep = "|"
    elif "\t" in sample:
        sep = "\t"
    else:
        sep = ","
    return pd.read_csv(path, sep=sep, dtype=str, engine="python", on_bad_lines="skip")


def county_name(row: pd.Series) -> str:
    if row.get("state_code") == "11":
        return "District of Columbia"
    if row.get("state_code") == "24":
        code = row.get("county_code", "")
        return MD_COUNTIES.get(code, f"Maryland county {code}".strip())
    return row.get("county_code", "")


def clean_coords(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["latitude", "longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        big = df[col].abs() > 180
        df.loc[big, col] = df.loc[big, col] / 1_000_000.0
    df.loc[df["longitude"] > 0, "longitude"] = -df.loc[df["longitude"] > 0, "longitude"]
    df.loc[~df["latitude"].between(35, 41), "latitude"] = pd.NA
    df.loc[~df["longitude"].between(-82, -74), "longitude"] = pd.NA
    return df


def read_metadata(path: Path) -> pd.DataFrame:
    df = read_delimited(path)
    cols = list(df.columns)
    st_col = detect_col(cols, ["state_code", "state", "fips_state_code", "fips_state"])
    sta_col = detect_col(cols, ["station_id", "station", "stationid", "station_number"])
    dir_col = detect_col(cols, ["direction", "travel_dir", "dir", "direction_of_travel"])
    lat_col = detect_col(cols, ["latitude", "lat"])
    lon_col = detect_col(cols, ["longitude", "lon", "long"])
    route_col = detect_col(cols, ["posted_signed_route", "route", "route_id", "route_name"])
    loc_col = detect_col(cols, ["station_location", "location_text", "location", "description"])
    county_col = detect_col(cols, ["county_code", "county", "cnty", "cntyfips"])
    src_col = detect_col(cols, ["source_file"])
    if not (st_col and sta_col and lat_col and lon_col):
        raise ValueError(f"Could not detect required metadata columns in {path}: {cols[:20]}")
    out = pd.DataFrame({
        "state_code": df[st_col].map(normalize_state),
        "station_id": df[sta_col].map(normalize_station),
        "direction": df[dir_col].map(normalize_code) if dir_col else "",
        "county_code": df[county_col].map(normalize_county) if county_col else "",
        "latitude": df[lat_col],
        "longitude": df[lon_col],
        "route": df[route_col].astype(str).str.strip() if route_col else "",
        "location_text": df[loc_col].astype(str).str.strip() if loc_col else "",
        "source_file": df[src_col].astype(str).str.strip() if src_col else path.name,
    })
    out = clean_coords(out)
    out["state_name"] = out["state_code"].map(STATE_NAMES).fillna(out["state_code"])
    out["county_name"] = out.apply(county_name, axis=1)
    return out


def summarize_hourly(path: Path) -> pd.DataFrame:
    hourly = pd.read_csv(path, dtype=str, low_memory=False)
    required = {"state_code", "station_id", "direction", "lane", "hour", "volume"}
    missing = required - set(hourly.columns)
    if missing:
        raise ValueError(f"Hourly file is missing required columns: {sorted(missing)}")
    hourly["state_code"] = hourly["state_code"].map(normalize_state)
    hourly["station_id"] = hourly["station_id"].map(normalize_station)
    hourly["direction"] = hourly["direction"].map(normalize_code)
    hourly["lane"] = hourly["lane"].map(normalize_code)
    hourly["volume"] = pd.to_numeric(hourly["volume"], errors="coerce").fillna(0)
    return (
        hourly.groupby(["state_code", "station_id", "direction"], as_index=False)
        .agg(
            lanes=("lane", lambda values: ",".join(sorted(set(map(str, values))))),
            n_lane_hour_records=("volume", "size"),
            total_volume=("volume", "sum"),
            max_hourly_lane_volume=("volume", "max"),
        )
        .assign(direction_label=lambda df: df["direction"].map(DIRECTION_LABELS).fillna(""))
    )


def join_used_to_metadata(used: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    meta = metadata.dropna(subset=["latitude", "longitude"]).copy()
    meta_dir = meta.drop_duplicates(["state_code", "station_id", "direction"])
    joined = used.merge(meta_dir, on=["state_code", "station_id", "direction"], how="left")
    no_coord = joined["latitude"].isna() | joined["longitude"].isna()
    if no_coord.any():
        meta_station = meta.drop_duplicates(["state_code", "station_id"])[[
            "state_code", "station_id", "county_code", "latitude", "longitude",
            "route", "location_text", "source_file", "state_name", "county_name",
        ]]
        fallback = used.loc[no_coord, ["state_code", "station_id"]].merge(meta_station, on=["state_code", "station_id"], how="left")
        for col in ["county_code", "latitude", "longitude", "route", "location_text", "source_file", "state_name", "county_name"]:
            joined.loc[no_coord, col] = fallback[col].to_numpy()
    joined["has_coordinates"] = joined["latitude"].notna() & joined["longitude"].notna()
    joined["county_name"] = joined.apply(county_name, axis=1)
    return joined


def in_bounds(df: pd.DataFrame, bounds: list[list[float]]) -> pd.Series:
    (lat_min, lon_min), (lat_max, lon_max) = bounds
    return df["latitude"].between(lat_min, lat_max) & df["longitude"].between(lon_min, lon_max)


def assign_screenlines(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mapped = joined[joined["has_coordinates"]].copy()
    for key, spec in SCREENLINE_DEFINITIONS.items():
        mask = in_bounds(mapped, spec["bounds"])
        mask &= mapped["state_code"].isin(spec["state_codes"])
        mask &= mapped["county_code"].fillna("").isin(spec["county_codes"])
        subset = mapped.loc[mask].copy()
        subset["screenline"] = key
        subset["screenline_label"] = spec["label"]
        subset["screenline_color"] = spec["color"]
        rows.append(subset)
    if not rows:
        return pd.DataFrame(columns=list(joined.columns) + ["screenline", "screenline_label", "screenline_color"])
    return pd.concat(rows, ignore_index=True).drop_duplicates(["screenline", "state_code", "station_id", "direction"])


def station_map_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in candidates.sort_values(["screenline", "state_code", "station_id", "direction"]).iterrows():
        label = f"{row.get('route', '')} {row.get('location_text', '')}".strip()
        rows.append({
            "include": 1,
            "screenline": row["screenline"],
            "state_code": row["state_code"],
            "station_id": row["station_id"],
            "direction": row["direction"],
            "lane": "",
            "station_label": label,
            "notes": f"Candidate for {row['screenline_label']}; county={row.get('county_name', '')}; lanes={row.get('lanes', '')}",
        })
    return pd.DataFrame(rows, columns=MAP_COLUMNS)


def feature_collection(df: pd.DataFrame) -> dict[str, Any]:
    features = []
    for _, row in df.iterrows():
        props = {
            "screenline": row.get("screenline", ""),
            "screenline_label": row.get("screenline_label", ""),
            "color": row.get("screenline_color", "#666666"),
            "state_code": row.get("state_code", ""),
            "county_name": row.get("county_name", ""),
            "station_id": row.get("station_id", ""),
            "direction": row.get("direction", ""),
            "direction_label": row.get("direction_label", ""),
            "lanes": row.get("lanes", ""),
            "route": row.get("route", ""),
            "location_text": row.get("location_text", ""),
            "total_volume": float(row.get("total_volume", 0) or 0),
            "records": int(row.get("n_lane_hour_records", 0) or 0),
        }
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(row["longitude"]), float(row["latitude"])]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}


def collection_bounds(features: list[dict[str, Any]]) -> list[list[float]]:
    if not features:
        return [[38.75, -77.25], [39.25, -76.75]]
    lats = [feature["geometry"]["coordinates"][1] for feature in features]
    lons = [feature["geometry"]["coordinates"][0] for feature in features]
    return [[min(lats), min(lons)], [max(lats), max(lons)]]


def write_leaflet_map(candidates: pd.DataFrame, all_joined: pd.DataFrame, out_path: Path) -> None:
    mapped_all = all_joined[all_joined["has_coordinates"]].copy()
    candidate_data = feature_collection(candidates)
    all_data = feature_collection(mapped_all)
    bounds = collection_bounds(candidate_data["features"] or all_data["features"])
    zones = [{"key": key, "label": spec["label"], "color": spec["color"], "bounds": spec["bounds"]} for key, spec in SCREENLINE_DEFINITIONS.items()]
    summary = candidates.groupby("screenline_label")["station_id"].nunique().to_dict() if not candidates.empty else {}
    html_text = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>FHWA Screenline Candidate Stations</title>
  <link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\">
  <style>
    html, body, #map {{ height: 100%; margin: 0; }}
    body {{ font-family: Arial, sans-serif; }}
    .panel {{ background: rgba(255,255,255,0.95); border: 1px solid #c8d0d8; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.2); max-width: 380px; padding: 10px 12px; }}
    .panel h1 {{ font-size: 15px; margin: 0 0 6px; }}
    .panel p, .panel li {{ font-size: 12px; line-height: 1.35; margin: 4px 0; }}
    .legend-row {{ align-items: center; display: flex; gap: 6px; }}
    .swatch {{ display: inline-block; height: 11px; width: 11px; }}
  </style>
</head>
<body>
<div id=\"map\"></div>
<script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>
<script>
const candidateData = {json.dumps(candidate_data, separators=(",", ":"))};
const allData = {json.dumps(all_data, separators=(",", ":"))};
const zones = {json.dumps(zones, separators=(",", ":"))};
const startBounds = {json.dumps(bounds)};
const summary = {json.dumps(summary, separators=(",", ":"))};
const map = L.map('map', {{ preferCanvas: true }});
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ maxZoom: 19, attribution: '&copy; OpenStreetMap contributors' }}).addTo(map);
map.fitBounds(startBounds, {{ padding: [24, 24] }});
function popupHtml(p) {{
  return `<strong>${{p.screenline_label || 'FHWA station'}}</strong><br>` +
    `Station: ${{p.state_code}} / ${{p.station_id}} / dir ${{p.direction}} ${{p.direction_label || ''}}<br>` +
    `County: ${{p.county_name || ''}}<br>` +
    `Route: ${{p.route || ''}}<br>` +
    `Location: ${{p.location_text || ''}}<br>` +
    `Lanes: ${{p.lanes || ''}}<br>` +
    `June 10 lane-volume sum: ${{Math.round(p.total_volume || 0).toLocaleString()}}<br>` +
    `Records: ${{(p.records || 0).toLocaleString()}}`;
}}
const allLayer = L.geoJSON(allData, {{
  pointToLayer: (feature, latlng) => L.circleMarker(latlng, {{ radius: 3, color: '#8b96a3', fillColor: '#8b96a3', fillOpacity: 0.45, weight: 1 }}),
  onEachFeature: (feature, layer) => {{ layer.bindPopup(popupHtml(feature.properties)); }}
}});
const candidateGroups = {{}};
for (const zone of zones) {{
  candidateGroups[zone.key] = L.layerGroup().addTo(map);
  L.rectangle(zone.bounds, {{ color: zone.color, weight: 1, fill: false, dashArray: '5,5' }}).bindTooltip(zone.label).addTo(map);
}}
L.geoJSON(candidateData, {{
  pointToLayer: (feature, latlng) => L.circleMarker(latlng, {{ radius: 7, color: feature.properties.color || '#333', fillColor: feature.properties.color || '#333', fillOpacity: 0.82, weight: 2 }}),
  onEachFeature: (feature, layer) => {{
    layer.bindTooltip(`${{feature.properties.screenline_label}}: ${{feature.properties.station_id}} dir ${{feature.properties.direction}}`);
    layer.bindPopup(popupHtml(feature.properties));
    const key = feature.properties.screenline;
    if (candidateGroups[key]) {{ candidateGroups[key].addLayer(layer); }}
  }}
}});
const overlays = {{ 'All mapped FHWA hourly stations': allLayer }};
for (const zone of zones) overlays[zone.label] = candidateGroups[zone.key];
L.control.layers(null, overlays, {{ collapsed: false }}).addTo(map);
const info = L.control({{ position: 'topright' }});
info.onAdd = () => {{
  const div = L.DomUtil.create('div', 'panel');
  const items = Object.entries(summary).map(([label, count]) => `<li>${{label}}: <strong>${{count}}</strong> stations</li>`).join('');
  div.innerHTML = `<h1>FHWA Screenline Candidate Stations</h1><p>Candidate station-direction points for DC-PG, PG-Montgomery, and Montgomery-DC review.</p><ul>${{items}}</ul><p>Rectangles are broad review zones, not official screenline geometry.</p>`;
  return div;
}};
info.addTo(map);
const legend = L.control({{ position: 'bottomright' }});
legend.onAdd = () => {{
  const div = L.DomUtil.create('div', 'panel');
  div.innerHTML = zones.map(z => `<div class=\"legend-row\"><span class=\"swatch\" style=\"background:${{z.color}}\"></span>${{z.label}}</div>`).join('');
  return div;
}};
legend.addTo(map);
</script>
</body>
</html>
"""
    out_path.write_text(html_text)


def choose_metadata_path(arg_path: str | None) -> Path:
    if arg_path:
        return Path(arg_path)
    if DEFAULT_METADATA_RAW.exists():
        return DEFAULT_METADATA_RAW
    return DEFAULT_METADATA_PARSED


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hourly", type=Path, default=DEFAULT_HOURLY)
    parser.add_argument("--station-metadata", default=None)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    metadata_path = choose_metadata_path(args.station_metadata)
    used = summarize_hourly(args.hourly)
    metadata = read_metadata(metadata_path)
    joined = join_used_to_metadata(used, metadata)
    candidates = assign_screenlines(joined)
    station_map = station_map_rows(candidates)
    joined_path = args.outdir / "fhwa_used_stations_joined_to_metadata.csv"
    candidates_path = args.outdir / "fhwa_screenline_station_candidates.csv"
    station_map_path = args.outdir / "fhwa_screenline_station_map_candidates.csv"
    missing_path = args.outdir / "fhwa_used_stations_missing_coordinates.csv"
    map_path = args.outdir / "fhwa_screenline_station_map.html"
    joined.to_csv(joined_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    station_map.to_csv(station_map_path, index=False)
    joined.loc[~joined["has_coordinates"]].to_csv(missing_path, index=False)
    write_leaflet_map(candidates, joined, map_path)
    print("Wrote:")
    for path in [joined_path, candidates_path, station_map_path, missing_path, map_path]:
        print(f"  {path}")
    if candidates.empty:
        print("No candidates found inside the configured review zones.")
    else:
        print("\nCandidate station counts:")
        print(candidates.groupby("screenline_label")["station_id"].nunique().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
