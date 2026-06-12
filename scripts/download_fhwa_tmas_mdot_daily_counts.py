#!/usr/bin/env python3
"""Download Maryland FHWA TMAS station-day and station-hour counts.

FHWA TMAS monthly volume files contain the continuous count records submitted by
state DOTs. Maryland permanent ATR station IDs appear as FHWA IDs such as
``0P0009``; the corresponding MDOT AADT Locator station ID is ``P0009``. This
script parses the fixed-width monthly ``.VOL`` records and writes both daily and
hourly validation inputs for the two-prong screenline validator.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

BASE_URL = "https://www.fhwa.dot.gov/policyinformation/tables/tmasdata"
MONTH_ABBR = {
    1: "jan",
    2: "feb",
    3: "mar",
    4: "apr",
    5: "may",
    6: "jun",
    7: "jul",
    8: "aug",
    9: "sep",
    10: "oct",
    11: "nov",
    12: "dec",
}


@dataclass(frozen=True)
class MonthRef:
    year: int
    month: int

    @property
    def volume_url(self) -> str:
        abbr = MONTH_ABBR[self.month]
        return f"{BASE_URL}/{self.year}/{abbr}_{self.year}_ccs_data.zip"

    @property
    def cache_name(self) -> str:
        return f"{self.year}_{self.month:02d}_{MONTH_ABBR[self.month]}_ccs_data.zip"


def month_refs(start: date, end: date) -> list[MonthRef]:
    refs: list[MonthRef] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        refs.append(MonthRef(y, m))
        m += 1
        if m == 13:
            y += 1
            m = 1
    return refs


def download(url: str, dest: Path, timeout: int = 180) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"}) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp.replace(dest)


def normalize_mdot_station_id(fhwa_station_id: str) -> str:
    value = str(fhwa_station_id).strip().upper()
    if len(value) == 6 and value.startswith("0") and value[1].isalpha():
        return value[1:]
    return value


def parse_fhwa_vol_line(line: str, allowed_states: set[str], start: date, end: date) -> dict[str, object] | None:
    line = line.rstrip("\r\n")
    if len(line) < 140 or line[0] != "3":
        return None
    state_code = line[1:3]
    if state_code not in allowed_states:
        return None
    try:
        year = 2000 + int(line[13:15])
        month = int(line[15:17])
        day = int(line[17:19])
        record_date = date(year, month, day)
    except ValueError:
        return None
    if record_date < start or record_date > end:
        return None

    hourly_field = line[20 : 20 + 24 * 5]
    if len(hourly_field) < 24 * 5:
        return None
    volumes: list[int] = []
    hourly_values: list[int | None] = []
    for offset in range(0, 24 * 5, 5):
        token = hourly_field[offset : offset + 5]
        if not token.isdigit():
            return None
        value = int(token)
        if value == 99999:
            hourly_values.append(None)
            continue
        volumes.append(value)
        hourly_values.append(value)

    fhwa_station_id = line[5:11].strip().upper()
    return {
        "state_code": state_code,
        "fhwa_station_id": fhwa_station_id,
        "station_id": normalize_mdot_station_id(fhwa_station_id),
        "direction": line[11:12].strip(),
        "lane": line[12:13].strip(),
        "date": record_date.isoformat(),
        "observed_count": sum(volumes),
        "hours_observed": len(volumes),
        "hourly_counts": "|".join("" if value is None else str(value) for value in hourly_values),
    }


def vol_member_names(zip_path: Path, state_abbrev: str = "MD") -> list[str]:
    prefix = state_abbrev.upper()
    with zipfile.ZipFile(zip_path) as archive:
        names = []
        for name in archive.namelist():
            basename = Path(name).name.upper()
            if Path(name).suffix.upper() != ".VOL":
                continue
            if basename.startswith(f"{prefix}_") or re.match(rf"^{prefix}\d{{4}}\.VOL$", basename):
                names.append(name)
    return names


def parse_month_zip(zip_path: Path, allowed_states: set[str], start: date, end: date, state_abbrev: str = "MD") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(zip_path) as archive:
        names = vol_member_names(zip_path, state_abbrev=state_abbrev)
        if not names:
            raise FileNotFoundError(f"No {state_abbrev}_*.VOL file found in {zip_path}")
        for name in names:
            with archive.open(name) as handle:
                for raw in handle:
                    line = raw.decode("latin1", errors="ignore")
                    row = parse_fhwa_vol_line(line, allowed_states, start, end)
                    if row is not None:
                        row["source_file"] = Path(name).name
                        rows.append(row)
    return rows


def station_data_url(year: int) -> str:
    return f"{BASE_URL}/{year}/{year}_station_data.zip"


def _microdegree(value: object, west_longitude: bool = False) -> float | None:
    try:
        number = float(str(value).strip())
    except Exception:
        return None
    if number == 0:
        return None
    if abs(number) > 180:
        number /= 1_000_000.0
    if west_longitude and number > 0:
        number = -number
    return number


def parse_station_zip(zip_path: Path, allowed_states: set[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith((".txt", ".csv")):
                continue
            with archive.open(name) as handle:
                text = handle.read().decode("latin1", errors="ignore")
            sample = text.splitlines()[0] if text.splitlines() else ""
            if "|" not in sample:
                continue
            reader = csv.DictReader(text.splitlines(), delimiter="|")
            for row in reader:
                state = str(row.get("State_Code", "")).strip().zfill(2)
                if state not in allowed_states:
                    continue
                fhwa_station_id = str(row.get("Station_Id", "")).strip().upper()
                rows.append(
                    {
                        "state_code": state,
                        "fhwa_station_id": fhwa_station_id,
                        "station_id": normalize_mdot_station_id(fhwa_station_id),
                        "direction": str(row.get("Travel_Dir", "")).strip(),
                        "latitude": _microdegree(row.get("Latitude")),
                        "longitude": _microdegree(row.get("Longitude"), west_longitude=True),
                        "f_system": str(row.get("F_System", "")).strip(),
                        "num_lanes_volume": str(row.get("Num_Lanes_Volume", "")).strip(),
                        "station_location": str(row.get("Station_Location", "")).strip(),
                        "posted_route_signing": str(row.get("Posted_Route_Signing", "")).strip(),
                        "posted_signed_route": str(row.get("Posted_Signed_Route", "")).strip(),
                        "year_record": str(row.get("Year_Record", "")).strip(),
                        "source_file": Path(name).name,
                    }
                )
    return rows


def aggregate_hourly(lane_rows: Iterable[dict[str, object]], min_hours: int) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in lane_rows:
        if int(row["hours_observed"]) < min_hours:
            continue
        values = str(row.get("hourly_counts", "")).split("|")
        if len(values) != 24:
            continue
        for hour, token in enumerate(values):
            if token == "":
                continue
            hourly_row = dict(row)
            hourly_row["hour"] = hour
            hourly_row["observed_count"] = int(token)
            grouped[(str(row["station_id"]), str(row["date"]), hour)].append(hourly_row)

    hourly_rows: list[dict[str, object]] = []
    for (station_id, day, hour), rows in grouped.items():
        hourly_rows.append(
            {
                "station_id": station_id,
                "date": day,
                "hour": hour,
                "observed_count": sum(int(row["observed_count"]) for row in rows),
                "source": "FHWA_TMAS_CCS_MD",
                "fhwa_station_ids": "|".join(sorted({str(row["fhwa_station_id"]) for row in rows})),
                "n_direction_lane_records": len(rows),
            }
        )
    return sorted(hourly_rows, key=lambda row: (str(row["station_id"]), str(row["date"]), int(row["hour"])))


def aggregate_daily(lane_rows: Iterable[dict[str, object]], min_hours: int) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in lane_rows:
        if int(row["hours_observed"]) >= min_hours:
            grouped[(str(row["station_id"]), str(row["date"]))].append(row)

    daily_rows: list[dict[str, object]] = []
    for (station_id, day), rows in grouped.items():
        daily_rows.append(
            {
                "station_id": station_id,
                "date": day,
                "observed_count": sum(int(row["observed_count"]) for row in rows),
                "source": "FHWA_TMAS_CCS_MD",
                "fhwa_station_ids": "|".join(sorted({str(row["fhwa_station_id"]) for row in rows})),
                "n_direction_lane_records": len(rows),
                "min_hours_observed": min(int(row["hours_observed"]) for row in rows),
                "max_hours_observed": max(int(row["hours_observed"]) for row in rows),
            }
        )
    return sorted(daily_rows, key=lambda row: (str(row["station_id"]), str(row["date"])))



def collapse_duplicate_direction_lane_rows(lane_rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Collapse repeated station/date/direction/lane submissions before station totals."""
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in lane_rows:
        key = (
            str(row["station_id"]),
            str(row["date"]),
            str(row["direction"]),
            str(row["lane"]),
            str(row["fhwa_station_id"]),
        )
        grouped[key].append(row)

    collapsed: list[dict[str, object]] = []
    for rows in grouped.values():
        chosen = max(rows, key=lambda row: int(row["observed_count"]))
        out = dict(chosen)
        out["source_file"] = "|".join(sorted({str(row["source_file"]) for row in rows}))
        out["source_record_count"] = len(rows)
        out["duplicate_policy"] = "max_observed_count_per_station_date_direction_lane"
        collapsed.append(out)
    return sorted(
        collapsed,
        key=lambda row: (
            str(row["station_id"]),
            str(row["date"]),
            str(row["direction"]),
            str(row["lane"]),
            str(row["fhwa_station_id"]),
        ),
    )

def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_date_arg(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2017-10-01")
    parser.add_argument("--end-date", default="2019-07-31")
    parser.add_argument("--state-code", action="append", default=["24"], help="FHWA state code to parse; default Maryland (24).")
    parser.add_argument("--state-abbrev", default="MD", help="Monthly .VOL filename prefix to parse.")
    parser.add_argument("--cache-dir", type=Path, default=Path(tempfile.gettempdir()) / "fhwa_tmas_raw")
    parser.add_argument("--out", type=Path, default=Path("data/external/mdot_daily_counts/station_daily_counts.csv"))
    parser.add_argument("--lane-audit-out", type=Path, default=Path("data/external/mdot_daily_counts/fhwa_tmas_direction_lane_daily_counts_2017-10-01_2019-07-31.csv"))
    parser.add_argument("--hourly-out", type=Path, default=Path("data/external/mdot_daily_counts/station_hourly_counts.csv"))
    parser.add_argument("--metadata-out", type=Path, default=Path("data/external/mdot_daily_counts/fhwa_tmas_station_metadata_2017_2019.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/external/mdot_daily_counts/fhwa_tmas_daily_counts_manifest.json"))
    parser.add_argument("--min-hours", type=int, default=24)
    args = parser.parse_args()

    start = parse_date_arg(args.start_date)
    end = parse_date_arg(args.end_date)
    allowed_states = {str(code).zfill(2) for code in args.state_code}

    monthly_rows: list[dict[str, object]] = []
    downloaded_volume_urls: list[str] = []
    for ref in month_refs(start, end):
        zip_path = args.cache_dir / ref.cache_name
        print(f"Fetching/parsing {ref.volume_url}", flush=True)
        download(ref.volume_url, zip_path)
        downloaded_volume_urls.append(ref.volume_url)
        rows = parse_month_zip(zip_path, allowed_states, start, end, state_abbrev=args.state_abbrev)
        monthly_rows.extend(rows)
        print(f"  parsed {len(rows)} direction/lane-day rows", flush=True)

    metadata_rows: list[dict[str, object]] = []
    downloaded_station_urls: list[str] = []
    for year in range(start.year, end.year + 1):
        url = station_data_url(year)
        zip_path = args.cache_dir / f"{year}_station_data.zip"
        print(f"Fetching/parsing {url}", flush=True)
        download(url, zip_path)
        downloaded_station_urls.append(url)
        metadata_rows.extend(parse_station_zip(zip_path, allowed_states))

    raw_direction_lane_days = len(monthly_rows)
    monthly_rows = collapse_duplicate_direction_lane_rows(monthly_rows)
    duplicate_direction_lane_groups = sum(1 for row in monthly_rows if int(row["source_record_count"]) > 1)
    print(
        "Collapsed "
        f"{raw_direction_lane_days} raw direction/lane-day rows to {len(monthly_rows)} unique rows "
        f"({duplicate_direction_lane_groups} duplicate groups)",
        flush=True,
    )

    daily_rows = aggregate_daily(monthly_rows, min_hours=args.min_hours)
    hourly_rows = aggregate_hourly(monthly_rows, min_hours=args.min_hours)
    write_csv(args.out, daily_rows)
    write_csv(args.hourly_out, hourly_rows)
    write_csv(args.lane_audit_out, sorted(monthly_rows, key=lambda row: (str(row["station_id"]), str(row["date"]), str(row["direction"]), str(row["lane"]))))
    write_csv(args.metadata_out, sorted(metadata_rows, key=lambda row: (str(row["station_id"]), str(row["direction"]), str(row["year_record"]))))

    manifest = {
        "source": "FHWA Traffic Monitoring Analysis System continuous count station monthly volume files",
        "source_base_url": BASE_URL,
        "volume_urls": downloaded_volume_urls,
        "station_data_urls": downloaded_station_urls,
        "pulled_at_utc": datetime.now(UTC).isoformat(),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "state_codes": sorted(allowed_states),
        "state_abbrev": args.state_abbrev,
        "output": str(args.out),
        "lane_audit_output": str(args.lane_audit_out),
        "hourly_output": str(args.hourly_out),
        "metadata_output": str(args.metadata_out),
        "station_days": len(daily_rows),
        "station_hours": len(hourly_rows),
        "stations": len({row["station_id"] for row in daily_rows}),
        "direction_lane_days": len(monthly_rows),
        "raw_direction_lane_days": raw_direction_lane_days,
        "duplicate_direction_lane_groups": duplicate_direction_lane_groups,
        "duplicate_direction_lane_policy": "max observed_count within identical station/date/direction/lane/FHWA station id",
        "min_hours_per_direction_lane": args.min_hours,
        "station_id_note": "FHWA station IDs like 0P0009 are normalized to MDOT AADT Locator IDs like P0009.",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(daily_rows)} station-day rows to {args.out}")
    print(f"Wrote {len(hourly_rows)} station-hour rows to {args.hourly_out}")


if __name__ == "__main__":
    main()
