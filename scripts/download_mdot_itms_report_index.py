#!/usr/bin/env python3
"""Download MDOT I-TMS count-report metadata for mapped validation stations.

The public MDOT I-TMS application exposes station count *report rows* through
an ASP.NET WebForms page. This script captures those official report rows for
the stations used by the AADT screenline validation. It intentionally does not
invent station-day totals; the Volume Detail report body is still a separate
ReportViewer workflow.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import html
import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

BASE_URL = "https://maps.roads.maryland.gov/itms_public/"
DEFAULT_START = "2017-10-01"
DEFAULT_END = "2019-07-31"


def _form_inputs(markup: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for tag in re.findall(r"<input\b[^>]*>", markup, flags=re.I):
        name = re.search(r'\bname="([^"]+)"', tag)
        if not name:
            continue
        value = re.search(r'\bvalue="([^"]*)"', tag)
        fields[html.unescape(name.group(1))] = html.unescape(value.group(1)) if value else ""
    return fields


def _clean_cell(cell: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<.*?>", "", cell)).strip()


def _parse_date(value: str) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, format="%m/%d/%Y", errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.normalize()


def _overlaps_window(start: pd.Timestamp | None, end: pd.Timestamp | None, window_start: pd.Timestamp, window_end: pd.Timestamp) -> bool:
    if start is None or end is None:
        return False
    return start <= window_end and end >= window_start


def load_station_ids(path: Path, column: str = "station_id") -> list[str]:
    frame = pd.read_parquet(path)
    if column not in frame.columns:
        raise ValueError(f"{path} does not contain required column {column!r}")
    return sorted({str(value).strip() for value in frame[column].dropna() if str(value).strip()})


@dataclass(frozen=True)
class FetchResult:
    station_id: str
    rows: list[dict[str, str]]
    error: str | None = None


def fetch_station_report_rows(station_id: str, window_start: pd.Timestamp, window_end: pd.Timestamp, timeout: int = 60) -> FetchResult:
    session = requests.Session()
    try:
        response = session.get(BASE_URL, timeout=timeout)
        response.raise_for_status()
        fields = _form_inputs(response.text)
        fields.update({"stationID": station_id, "fromLink": "1", "btnReports": ""})
        response = session.post(BASE_URL, data=fields, timeout=timeout)
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - network error path
        return FetchResult(station_id=station_id, rows=[], error=str(exc))

    rows: list[dict[str, str]] = []
    for row_html in re.findall(r'<tr class="Grid(?:Row|AlternatingRow)Style">(.*?)</tr>', response.text, flags=re.S):
        cells = [_clean_cell(cell) for cell in re.findall(r"<td[^>]*>(.*?)</td>", row_html, flags=re.S)]
        if len(cells) < 14:
            continue

        start = _parse_date(cells[2])
        end = _parse_date(cells[3])
        if not _overlaps_window(start, end, window_start, window_end):
            continue

        station_match = re.search(r'hfStationID"[^>]+value="([^"]+)"', row_html)
        source_match = re.search(r'hfReportSource"[^>]+value="([^"]+)"', row_html)
        options = [
            {"value": html.unescape(value), "label": _clean_cell(label)}
            for value, label in re.findall(r'<option(?:[^>]*) value="([^"]*)"[^>]*>(.*?)</option>', row_html, flags=re.S)
            if value != "-1"
        ]
        report_station_id = html.unescape(station_match.group(1)) if station_match else ""
        rows.append(
            {
                "requested_station_id": station_id,
                "report_station_id": report_station_id,
                "location": cells[1],
                "start_date": start.date().isoformat() if start is not None else "",
                "end_date": end.date().isoformat() if end is not None else "",
                "day_of_week": cells[4],
                "duration": cells[5],
                "report_type_id": cells[0],
                "count_id": cells[6],
                "source": cells[7],
                "report_record_id": cells[13],
                "report_source": html.unescape(source_match.group(1)) if source_match else "",
                "report_options": json.dumps(options, sort_keys=True),
                "report_values": "|".join(option["value"] for option in options),
                "itms_station_url": f"{BASE_URL}?stationid={station_id}",
            }
        )
    return FetchResult(station_id=station_id, rows=rows)


def _write_csv(path: Path, rows: Iterable[dict[str, str]]) -> int:
    rows = list(rows)
    fieldnames = [
        "requested_station_id",
        "report_station_id",
        "location",
        "start_date",
        "end_date",
        "day_of_week",
        "duration",
        "report_type_id",
        "count_id",
        "source",
        "report_record_id",
        "report_source",
        "report_values",
        "report_options",
        "itms_station_url",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station-map", type=Path, default=Path("outputs/runs/quick_test/geo/screenline_station_map.parquet"))
    parser.add_argument("--out", type=Path, default=Path("data/external/mdot_daily_counts/itms_station_report_index_2017-10-01_2019-07-31.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/external/mdot_daily_counts/itms_station_report_index_manifest.json"))
    parser.add_argument("--start-date", default=DEFAULT_START)
    parser.add_argument("--end-date", default=DEFAULT_END)
    parser.add_argument("--limit", type=int, default=0, help="Optional station limit for smoke testing.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    stations = load_station_ids(args.station_map)
    if args.limit:
        stations = stations[: args.limit]
    window_start = pd.Timestamp(args.start_date)
    window_end = pd.Timestamp(args.end_date)

    all_rows: list[dict[str, str]] = []
    errors: dict[str, str] = {}
    started = time.time()
    with futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(fetch_station_report_rows, station, window_start, window_end): station
            for station in stations
        }
        for i, future in enumerate(futures.as_completed(future_map), start=1):
            result = future.result()
            if result.error:
                errors[result.station_id] = result.error
            all_rows.extend(result.rows)
            if i % 100 == 0 or i == len(future_map):
                print(f"Fetched {i}/{len(future_map)} stations; matched rows={len(all_rows)}; errors={len(errors)}", flush=True)

    row_count = _write_csv(args.out, all_rows)
    manifest = {
        "source": "MDOT SHA Internet Traffic Monitoring System (I-TMS)",
        "source_url": BASE_URL,
        "station_map": str(args.station_map),
        "output": str(args.out),
        "pulled_at_utc": datetime.now(UTC).isoformat(),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "stations_requested": len(stations),
        "report_rows": row_count,
        "errors": errors,
        "note": "This file indexes official I-TMS count report rows. It is not a station-day observed-count table.",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {row_count} rows to {args.out} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
