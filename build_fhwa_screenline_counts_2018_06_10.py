#!/usr/bin/env python3
"""
Build FHWA TMAS observed hourly screenline-count CSV for June 10, 2018.

What it does:
  1) Downloads FHWA 2018 station data + June 2018 CCS hourly volume data.
  2) Extracts the ZIPs.
  3) Converts June 10, 2018 hourly volume records to a long station-hour table.
  4) Joins a user-provided station-to-screenline map.
  5) Sums station/lane/direction volumes to hourly screenline counts.

Why the station-to-screenline map is explicit:
  FHWA TMAS gives count-station records, not TPB screenline definitions. For a
  defensible validation, you should explicitly select which stations belong to
  each screenline. The script writes an all-station hourly audit CSV so you can
  fill the map quickly if automatic metadata parsing is insufficient.

Run:
  python build_fhwa_screenline_counts_2018_06_10.py

Outputs:
  outputs/fhwa_2018_06_10_all_station_hourly.csv
  outputs/fhwa_2018_06_10_station_metadata_parsed.csv
  outputs/fhwa_2018_06_10_screenline_hourly_counts.csv   [if station map has rows]
  outputs/fhwa_2018_06_10_screenline_daily_counts.csv    [if station map has rows]

Inputs created/used:
  screenline_station_map.csv

Source URLs:
  https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/2018/2018_station_data.zip
  https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/2018/jun_2018_ccs_data.zip
"""
from __future__ import annotations

import csv
import io
import os
import re
import sys
import zipfile
import shutil
from pathlib import Path
from typing import Iterable, Optional
from urllib.request import Request, urlopen

import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "fhwa_tmas_raw"
OUT = BASE / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

STATION_ZIP_URL = "https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/2018/2018_station_data.zip"
JUNE_ZIP_URL = "https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/2018/jun_2018_ccs_data.zip"
TARGET_DATE = "2018-06-10"
TARGET_PATTERNS = ["20180610", "180610", "06102018", "061018"]

# These are the states needed for the DC regional screenlines.
REGION_STATE_CODES = {"11", "24", "51"}  # DC, MD, VA

MAP_PATH = BASE / "screenline_station_map.csv"

MAP_COLUMNS = [
    "include",
    "screenline",
    "state_code",
    "station_id",
    "direction",
    "lane",
    "station_label",
    "notes",
]

SCREENLINE_EXAMPLES = [
    ["", "dc_cordon", "", "", "", "", "", "Fill state_code/station_id after reviewing all_station_hourly + metadata"],
    ["", "dc_md", "", "", "", "", "", "Maryland/DC crossings or boundary counters"],
    ["", "dc_va_potomac", "", "", "", "", "", "Potomac River crossings between VA and DC"],
    ["", "montgomery_dc", "", "", "", "", "", "Montgomery County/DC crossings"],
    ["", "pg_dc", "", "", "", "", "", "Prince George's County/DC crossings"],
    ["", "nova_dc", "", "", "", "", "", "Northern Virginia/DC crossings"],
]


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Already downloaded: {dest.name}")
        return
    print(f"Downloading {url}")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=180) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    print(f"Saved {dest} ({dest.stat().st_size:,} bytes)")


def unzip(zip_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    files = [p for p in out_dir.rglob("*") if p.is_file()]
    print(f"Extracted {len(files)} files from {zip_path.name}")
    return files


def read_text_lines(path: Path) -> list[str]:
    raw = path.read_bytes()
    for enc in ["utf-8", "latin1", "cp1252"]:
        try:
            return raw.decode(enc).splitlines()
        except UnicodeDecodeError:
            continue
    return raw.decode("latin1", errors="ignore").splitlines()


def try_delimited_table(path: Path) -> Optional[pd.DataFrame]:
    """Try reading a file as comma/pipe/tab delimited; return None if unsuitable."""
    try:
        sample = path.read_text(encoding="latin1", errors="ignore")[:5000]
        if not any(sep in sample for sep in [",", "|", "\t"]):
            return None
        sep = None
        if "|" in sample:
            sep = "|"
        elif "\t" in sample:
            sep = "\t"
        else:
            sep = ","
        df = pd.read_csv(path, sep=sep, dtype=str, low_memory=False)
        if df.shape[1] < 4:
            return None
        return df
    except Exception:
        return None


def normalize_state(x: object) -> str:
    s = "" if pd.isna(x) else str(x).strip()
    s = re.sub(r"\D", "", s)
    return s.zfill(2)[-2:] if s else ""


def normalize_station(x: object) -> str:
    s = "" if pd.isna(x) else str(x).strip()
    return s.strip().upper()


def normalize_direction(x: object) -> str:
    s = "" if pd.isna(x) else str(x).strip().upper()
    return s


def normalize_lane(x: object) -> str:
    s = "" if pd.isna(x) else str(x).strip().upper()
    s = re.sub(r"\.0$", "", s)
    return s


def detect_col(cols: Iterable[str], candidates: list[str]) -> Optional[str]:
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


def parse_station_metadata(files: list[Path]) -> pd.DataFrame:
    rows = []
    for path in files:
        # Skip very large volume files accidentally included.
        if path.stat().st_size > 100_000_000:
            continue
        df_delim = try_delimited_table(path)
        if df_delim is not None:
            cols = list(df_delim.columns)
            st_col = detect_col(cols, ["state", "state_code", "fips_state_code", "fips_state"])
            sta_col = detect_col(cols, ["station", "station_id", "stationid", "station_number"])
            dir_col = detect_col(cols, ["direction", "dir", "direction_of_travel"])
            lat_col = detect_col(cols, ["latitude", "lat"])
            lon_col = detect_col(cols, ["longitude", "lon", "long"])
            route_col = detect_col(cols, ["route", "route_id", "route_name"])
            loc_col = detect_col(cols, ["location", "station_location", "description"])
            if st_col and sta_col:
                tmp = pd.DataFrame({
                    "state_code": df_delim[st_col].map(normalize_state),
                    "station_id": df_delim[sta_col].map(normalize_station),
                    "direction": df_delim[dir_col].map(normalize_direction) if dir_col else "",
                    "latitude": pd.to_numeric(df_delim[lat_col], errors="coerce") if lat_col else pd.NA,
                    "longitude": pd.to_numeric(df_delim[lon_col], errors="coerce") if lon_col else pd.NA,
                    "route": df_delim[route_col].astype(str) if route_col else "",
                    "location_text": df_delim[loc_col].astype(str) if loc_col else "",
                    "source_file": path.name,
                })
                rows.append(tmp)
                continue

        # Fixed-width fallback: collect state, station, direction, and any plausible decimal lat/lon.
        for line in read_text_lines(path):
            if len(line.strip()) < 8:
                continue
            # Many TMG station records start with record type + 2-digit state.
            state = line[1:3] if len(line) >= 3 and line[1:3].isdigit() else ""
            if state not in REGION_STATE_CODES:
                # Try first standalone two-digit code in the line.
                mstate = re.search(r"\b(11|24|51)\b", line[:20])
                state = mstate.group(1) if mstate else state
            if state not in REGION_STATE_CODES:
                continue

            # Station ID guess: chars 3:9 is common in TMG-like fixed files.
            station = line[3:9].strip() if len(line) >= 9 else ""
            direction = line[9:10].strip() if len(line) >= 10 else ""

            nums = re.findall(r"[-+]?\d+\.\d+", line)
            lat = lon = None
            for n in nums:
                v = float(n)
                if 37.0 <= v <= 40.5 and lat is None:
                    lat = v
                if -80.0 <= v <= -74.0 and lon is None:
                    lon = v
            if station:
                rows.append(pd.DataFrame([{
                    "state_code": normalize_state(state),
                    "station_id": normalize_station(station),
                    "direction": normalize_direction(direction),
                    "latitude": lat,
                    "longitude": lon,
                    "route": "",
                    "location_text": line[10:120].strip() if len(line) > 10 else "",
                    "source_file": path.name,
                }]))
    if not rows:
        return pd.DataFrame(columns=["state_code", "station_id", "direction", "latitude", "longitude", "route", "location_text", "source_file"])
    out = pd.concat(rows, ignore_index=True).drop_duplicates()
    out = out[out["state_code"].isin(REGION_STATE_CODES)].copy()
    return out


def parse_volume_delimited(path: Path) -> Optional[pd.DataFrame]:
    df = try_delimited_table(path)
    if df is None:
        return None
    cols = list(df.columns)
    state_col = detect_col(cols, ["state", "state_code", "fips_state_code"])
    station_col = detect_col(cols, ["station_id", "station", "stationid"])
    dir_col = detect_col(cols, ["direction", "dir", "direction_of_travel"])
    lane_col = detect_col(cols, ["lane", "lane_of_travel"])
    year_col = detect_col(cols, ["year"])
    month_col = detect_col(cols, ["month"])
    day_col = detect_col(cols, ["day", "day_of_month"])
    hour_col = detect_col(cols, ["hour_record", "hour", "hourrecord"])
    vol_col = detect_col(cols, ["hour_volume", "volume", "traffic_volume", "hourvolume"])

    if not (state_col and station_col and vol_col):
        return None

    if year_col and month_col and day_col:
        date = pd.to_datetime(dict(
            year=pd.to_numeric(df[year_col], errors="coerce"),
            month=pd.to_numeric(df[month_col], errors="coerce"),
            day=pd.to_numeric(df[day_col], errors="coerce"),
        ), errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        date_col = detect_col(cols, ["date", "record_date"])
        if not date_col:
            return None
        date = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")

    if hour_col:
        hour = pd.to_numeric(df[hour_col], errors="coerce")
        # FHWA tools sometimes use 1..24; convert to 0..23 if needed.
        if hour.dropna().between(1, 24).all():
            hour = hour - 1
    else:
        # Wide hourly columns fallback.
        hcols = []
        for c in cols:
            cc = re.sub(r"[^A-Z0-9]", "", c.upper())
            if re.fullmatch(r"(HOUR|HR|VOL|VOLUME)?0?([1-9]|1[0-9]|2[0-4])", cc):
                hcols.append(c)
        if len(hcols) < 24:
            return None
        keep = df.assign(_date=date)
        keep = keep[keep["_date"] == TARGET_DATE]
        long = []
        for idx, c in enumerate(hcols[:24]):
            tmp = keep[[state_col, station_col]].copy()
            tmp["direction"] = keep[dir_col] if dir_col else ""
            tmp["lane"] = keep[lane_col] if lane_col else ""
            tmp["date"] = TARGET_DATE
            tmp["hour"] = idx
            tmp["volume"] = pd.to_numeric(keep[c], errors="coerce")
            long.append(tmp)
        out = pd.concat(long, ignore_index=True)
        out.rename(columns={state_col: "state_code", station_col: "station_id"}, inplace=True)
        out["state_code"] = out["state_code"].map(normalize_state)
        out["station_id"] = out["station_id"].map(normalize_station)
        out["direction"] = out["direction"].map(normalize_direction)
        out["lane"] = out["lane"].map(normalize_lane)
        return out

    out = pd.DataFrame({
        "state_code": df[state_col].map(normalize_state),
        "station_id": df[station_col].map(normalize_station),
        "direction": df[dir_col].map(normalize_direction) if dir_col else "",
        "lane": df[lane_col].map(normalize_lane) if lane_col else "",
        "date": date,
        "hour": hour,
        "volume": pd.to_numeric(df[vol_col], errors="coerce"),
    })
    out = out[out["date"] == TARGET_DATE]
    return out


def parse_volume_fixed_width(path: Path) -> pd.DataFrame:
    """
    Heuristic parser for pre-2020 FHWA fixed-width continuous count data.

    It finds records containing the target date and takes the last 24 5-digit
    numeric groups as hourly volumes. State/station/direction/lane are inferred
    from common TMG-like leading fields. If your file layout differs, use FHWA's
    official rearrangement tool first, then put the converted CSV in the raw folder.
    """
    rows = []
    for line in read_text_lines(path):
        clean = line.rstrip("\n\r")
        compact = re.sub(r"\s+", "", clean)
        if not any(pat in compact for pat in TARGET_PATTERNS):
            continue

        # Common leading layout:
        # record type, state (2), functional class, facility type, station id (6),
        # direction, lane, year/month/day...
        state = clean[1:3] if len(clean) >= 3 and clean[1:3].isdigit() else ""
        station = clean[5:11].strip() if len(clean) >= 11 else ""
        direction = clean[11:12].strip() if len(clean) >= 12 else ""
        lane = clean[12:13].strip() if len(clean) >= 13 else ""

        # Robust fallback: find state at start-ish and target date position.
        if normalize_state(state) not in REGION_STATE_CODES:
            m = re.match(r"^\D?(11|24|51)([A-Z0-9]{3,12})", compact)
            if m:
                state = m.group(1)
                station = station or m.group(2)[:6]

        state = normalize_state(state)
        if state not in REGION_STATE_CODES:
            continue

        # Pull numeric groups. Volumes are commonly 24 five-digit fields.
        groups5 = re.findall(r"\d{5}", clean)
        if len(groups5) >= 24:
            vols = [int(x) for x in groups5[-24:]]
        else:
            # Last resort: collect all ints near the line end and take last 24 plausible volumes.
            ints = [int(x) for x in re.findall(r"\d+", clean)]
            if len(ints) < 24:
                continue
            vols = ints[-24:]

        for h, v in enumerate(vols):
            rows.append({
                "state_code": state,
                "station_id": normalize_station(station),
                "direction": normalize_direction(direction),
                "lane": normalize_lane(lane),
                "date": TARGET_DATE,
                "hour": h,
                "volume": v,
                "source_file": path.name,
                "raw_key_guess": compact[:40],
            })
    return pd.DataFrame(rows)


def parse_volume_files(files: list[Path]) -> pd.DataFrame:
    frames = []
    for path in files:
        if path.suffix.lower() in {".pdf", ".doc", ".docx", ".xlsx"}:
            continue
        if path.stat().st_size == 0:
            continue
        delim = parse_volume_delimited(path)
        if delim is not None and len(delim):
            delim["source_file"] = path.name
            frames.append(delim)
            continue
        fixed = parse_volume_fixed_width(path)
        if len(fixed):
            frames.append(fixed)
    if not frames:
        return pd.DataFrame(columns=["state_code", "station_id", "direction", "lane", "date", "hour", "volume", "source_file"])
    out = pd.concat(frames, ignore_index=True)
    out = out[out["state_code"].isin(REGION_STATE_CODES)].copy()
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out = out.dropna(subset=["volume"])
    out["hour"] = pd.to_numeric(out["hour"], errors="coerce").astype("Int64")
    out = out[out["hour"].between(0, 23)]
    return out


def create_map_template() -> None:
    if MAP_PATH.exists():
        return
    with open(MAP_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(MAP_COLUMNS)
        writer.writerows(SCREENLINE_EXAMPLES)
    print(f"Created template: {MAP_PATH.name}")


def aggregate_with_map(hourly: pd.DataFrame, meta: pd.DataFrame) -> None:
    create_map_template()
    smap = pd.read_csv(MAP_PATH, dtype=str).fillna("")
    if "include" not in smap.columns:
        smap["include"] = ""
    smap = smap[smap["include"].str.strip().str.lower().isin(["1", "y", "yes", "true", "x"])]
    if smap.empty:
        print("screenline_station_map.csv has no included rows yet; skipping screenline aggregation.")
        print("Fill include=1, screenline, state_code, station_id, and optional direction/lane, then rerun.")
        return

    for c in ["state_code", "station_id", "direction", "lane"]:
        if c not in smap.columns:
            smap[c] = ""
    smap["state_code"] = smap["state_code"].map(normalize_state)
    smap["station_id"] = smap["station_id"].map(normalize_station)
    smap["direction"] = smap["direction"].map(normalize_direction)
    smap["lane"] = smap["lane"].map(normalize_lane)

    rows = []
    for _, m in smap.iterrows():
        subset = hourly[(hourly["state_code"] == m["state_code"]) & (hourly["station_id"] == m["station_id"])]
        if m["direction"]:
            subset = subset[subset["direction"] == m["direction"]]
        if m["lane"]:
            subset = subset[subset["lane"] == m["lane"]]
        if subset.empty:
            continue
        subset = subset.copy()
        subset["screenline"] = m["screenline"]
        subset["station_label"] = m.get("station_label", "")
        rows.append(subset)

    if not rows:
        print("No hourly records matched the included station map rows; check station IDs/directions/lanes.")
        return

    matched = pd.concat(rows, ignore_index=True)
    matched_path = OUT / "fhwa_2018_06_10_screenline_matched_station_hourly.csv"
    matched.to_csv(matched_path, index=False)

    hourly_counts = (
        matched.groupby(["screenline", "date", "hour"], as_index=False)
        .agg(observed_count=("volume", "sum"), n_station_lane_records=("volume", "size"))
        .sort_values(["screenline", "hour"])
    )
    hourly_path = OUT / "fhwa_2018_06_10_screenline_hourly_counts.csv"
    hourly_counts.to_csv(hourly_path, index=False)

    daily = (
        hourly_counts.groupby(["screenline", "date"], as_index=False)
        .agg(observed_daily_count=("observed_count", "sum"), n_hourly_rows=("observed_count", "size"))
    )
    daily_path = OUT / "fhwa_2018_06_10_screenline_daily_counts.csv"
    daily.to_csv(daily_path, index=False)
    print(f"Wrote {hourly_path}")
    print(f"Wrote {daily_path}")


def main() -> int:
    station_zip = DATA / "2018_station_data.zip"
    june_zip = DATA / "jun_2018_ccs_data.zip"
    download(STATION_ZIP_URL, station_zip)
    download(JUNE_ZIP_URL, june_zip)

    station_files = unzip(station_zip, DATA / "station_2018")
    june_files = unzip(june_zip, DATA / "jun_2018")

    meta = parse_station_metadata(station_files)
    meta_path = OUT / "fhwa_2018_06_10_station_metadata_parsed.csv"
    meta.to_csv(meta_path, index=False)
    print(f"Wrote {meta_path} ({len(meta):,} metadata rows)")

    hourly = parse_volume_files(june_files)
    if hourly.empty:
        print("ERROR: Could not parse hourly volume records. Use FHWA's official rearrangement tool first, then place the converted CSV in fhwa_tmas_raw/jun_2018 and rerun.", file=sys.stderr)
        return 2

    # Add metadata where possible.
    hourly["state_code"] = hourly["state_code"].map(normalize_state)
    hourly["station_id"] = hourly["station_id"].map(normalize_station)
    hourly["direction"] = hourly["direction"].map(normalize_direction)
    hourly["lane"] = hourly["lane"].map(normalize_lane)
    hourly = hourly.sort_values(["state_code", "station_id", "direction", "lane", "hour"])

    all_path = OUT / "fhwa_2018_06_10_all_station_hourly.csv"
    hourly.to_csv(all_path, index=False)
    print(f"Wrote {all_path} ({len(hourly):,} hourly rows)")

    # Candidate audit: stations with daily totals and metadata.
    daily_station = (
        hourly.groupby(["state_code", "station_id", "direction", "lane"], as_index=False)
        .agg(daily_count=("volume", "sum"), hours_observed=("hour", "nunique"))
    )
    if not meta.empty:
        meta_small = meta.drop_duplicates(["state_code", "station_id", "direction"])
        daily_station = daily_station.merge(meta_small, on=["state_code", "station_id", "direction"], how="left")
    audit_path = OUT / "fhwa_2018_06_10_station_daily_audit.csv"
    daily_station.sort_values(["state_code", "daily_count"], ascending=[True, False]).to_csv(audit_path, index=False)
    print(f"Wrote {audit_path}")

    aggregate_with_map(hourly, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
