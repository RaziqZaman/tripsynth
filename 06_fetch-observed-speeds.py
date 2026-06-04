#!/usr/bin/env python3
"""Fetch observed VDOT 511 travel-time segment speeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.request import Request, urlopen


DEFAULT_RESPONSE_URL = (
    "https://data.511-atis-ttrip-prod.iteriscloud.com/"
    "datasets/travelTime/travel_time_response.json"
)
DEFAULT_METADATA_URL = (
    "https://data.511-atis-ttrip-prod.iteriscloud.com/"
    "datasets/travelTime/travel_time_segments_metadata.json"
)
DEFAULT_OUT = Path("06x_stage1_osm/observed_speeds_raw.csv")
DIRECTION_WORDS = {
    "n", "s", "e", "w", "nb", "sb", "eb", "wb",
    "north", "south", "east", "west",
    "northbound", "southbound", "eastbound", "westbound",
    "inner", "outer", "both", "direction1", "direction2",
}
SPEED_KEYS = ("avgSpeed", "averageSpeed", "currentSpeed", "speed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-url", default=DEFAULT_RESPONSE_URL)
    parser.add_argument("--metadata-url", default=DEFAULT_METADATA_URL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--min-speed-mph", type=float, default=1.0)
    parser.add_argument("--include-closed", action="store_true")
    return parser.parse_args()


def fetch_json(url: str, timeout: float) -> Any:
    request = Request(url, headers={"User-Agent": "tripsynth-speed-validation/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def scalar_items(record: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in record.items()
        if not isinstance(value, (dict, list))
    }


def is_direction(value: str) -> bool:
    return value.strip().lower().replace(" ", "") in DIRECTION_WORDS


def path_segment_id(path: tuple[str, ...]) -> str:
    for part in reversed(path):
        clean = part.strip()
        if clean and not is_direction(clean) and any(ch.isdigit() for ch in clean):
            return clean
    for part in reversed(path):
        clean = part.strip()
        if clean and not is_direction(clean):
            return clean
    return path[-1] if path else ""


def path_direction(path: tuple[str, ...], record: dict[str, Any]) -> str:
    for key in ("direction", "dir", "Direction", "Dir"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    for part in reversed(path):
        if is_direction(part):
            return part
    return ""


def has_speed(record: dict[str, Any]) -> bool:
    lower = {str(key).lower() for key in record}
    return any(key.lower() in lower for key in SPEED_KEYS)


def iter_speed_records(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], dict[str, Any]]]:
    if isinstance(value, dict):
        if has_speed(value):
            yield path, scalar_items(value)
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                yield from iter_speed_records(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_speed_records(child, path + (str(index),))


def iter_metadata_records(
    value: Any,
    path: tuple[str, ...] = (),
    inherited: dict[str, Any] | None = None,
) -> Iterator[tuple[tuple[str, ...], dict[str, Any]]]:
    inherited = dict(inherited or {})
    if isinstance(value, dict):
        current = inherited | scalar_items(value)
        children = [(key, child) for key, child in value.items() if isinstance(child, (dict, list))]
        if not children:
            yield path, current
            return
        if any(field_value(current, *COORD_KEYS) for COORD_KEYS in [START_LAT_KEYS, END_LAT_KEYS, START_LON_KEYS, END_LON_KEYS]):
            yield path, current
        for key, child in children:
            yield from iter_metadata_records(child, path + (str(key),), current)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_metadata_records(child, path + (str(index),), inherited)


def normalize_key(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def field_value(record: dict[str, Any], *names: str) -> str:
    if not record:
        return ""
    by_key = {normalize_key(str(key)): value for key, value in record.items()}
    for name in names:
        value = by_key.get(normalize_key(name))
        if value not in (None, ""):
            return str(value)
    return ""


def parse_float(value: Any, default: float = math.nan) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


START_LAT_KEYS = ("startLatitude", "start_latitude", "startLat", "fromLat", "beginLat", "lat1")
START_LON_KEYS = ("startLongitude", "start_longitude", "startLon", "fromLon", "beginLon", "lon1", "lng1")
END_LAT_KEYS = ("endLatitude", "end_latitude", "endLat", "toLat", "finishLat", "lat2")
END_LON_KEYS = ("endLongitude", "end_longitude", "endLon", "toLon", "finishLon", "lon2", "lng2")


def metadata_index(metadata: Any) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    by_segment: dict[str, dict[str, Any]] = {}
    by_segment_direction: dict[tuple[str, str], dict[str, Any]] = {}
    for path, row in iter_metadata_records(metadata):
        segment_id = field_value(row, "id", "segmentId", "segment_id", "travelTimeSegmentId") or path_segment_id(path)
        direction = path_direction(path, row).lower()
        if segment_id and segment_id not in by_segment:
            by_segment[segment_id] = row
        if segment_id and direction:
            by_segment_direction[(segment_id, direction)] = row
    return by_segment, by_segment_direction


def enriched_row(
    path: tuple[str, ...],
    speed: dict[str, Any],
    by_segment: dict[str, dict[str, Any]],
    by_segment_direction: dict[tuple[str, str], dict[str, Any]],
    fetched_at: str,
    response_url: str,
    metadata_url: str,
) -> dict[str, str]:
    segment_id = field_value(speed, "id", "segmentId", "segment_id", "travelTimeSegmentId") or path_segment_id(path)
    direction = path_direction(path, speed)
    metadata = by_segment_direction.get((segment_id, direction.lower()), by_segment.get(segment_id, {}))
    merged = dict(metadata) | speed
    observed_speed = parse_float(field_value(merged, *SPEED_KEYS))
    travel_time = parse_float(field_value(merged, "travelTime", "currentTime", "travel_time"))
    speed_time = parse_float(field_value(merged, "speedTime", "freeFlowTime", "normalTime"))
    length_miles = parse_float(field_value(merged, "length", "distance", "lengthMiles", "miles"))
    return {
        "segment_id": segment_id,
        "segment_group": path[0] if path else "",
        "direction": direction,
        "route": field_value(merged, "route", "routeName", "roadwayName", "roadName", "name"),
        "route_type": field_value(merged, "route_type", "routeType", "class", "roadClass"),
        "description": field_value(merged, "description", "displayName", "label"),
        "observed_speed_mph": f"{observed_speed:.8f}" if not math.isnan(observed_speed) else "",
        "travel_time_min": f"{travel_time:.8f}" if not math.isnan(travel_time) else "",
        "speed_time_min": f"{speed_time:.8f}" if not math.isnan(speed_time) else "",
        "length_miles": f"{length_miles:.8f}" if not math.isnan(length_miles) else "",
        "status": field_value(merged, "status"),
        "congestion": field_value(merged, "congestion", "congestionLevel"),
        "imputation": field_value(merged, "imputation", "isImputed"),
        "source_link_id": field_value(merged, "linkId", "link_id"),
        "start_lon": field_value(merged, *START_LON_KEYS),
        "start_lat": field_value(merged, *START_LAT_KEYS),
        "end_lon": field_value(merged, *END_LON_KEYS),
        "end_lat": field_value(merged, *END_LAT_KEYS),
        "start_milepost": field_value(merged, "startMilePost", "start_mile_post", "startMp"),
        "end_milepost": field_value(merged, "endMilePost", "end_mile_post", "endMp"),
        "fetched_at_utc": fetched_at,
        "source_response_url": response_url,
        "source_metadata_url": metadata_url,
    }


def keep_row(row: dict[str, str], min_speed_mph: float, include_closed: bool) -> bool:
    speed = parse_float(row.get("observed_speed_mph"), default=math.nan)
    if math.isnan(speed) or speed < min_speed_mph:
        return False
    status = row.get("status", "").strip().lower()
    if not include_closed and status in {"closed", "closure", "blocked"}:
        return False
    return True


def main() -> int:
    args = parse_args()
    response = fetch_json(args.response_url, args.timeout)
    metadata = fetch_json(args.metadata_url, args.timeout)
    by_segment, by_segment_direction = metadata_index(metadata)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rows = []
    for path, speed in iter_speed_records(response):
        row = enriched_row(
            path,
            speed,
            by_segment,
            by_segment_direction,
            fetched_at,
            args.response_url,
            args.metadata_url,
        )
        if keep_row(row, args.min_speed_mph, args.include_closed):
            rows.append(row)

    fieldnames = [
        "segment_id", "segment_group", "direction", "route", "route_type", "description",
        "observed_speed_mph", "travel_time_min", "speed_time_min", "length_miles",
        "status", "congestion", "imputation", "source_link_id",
        "start_lon", "start_lat", "end_lon", "end_lat", "start_milepost", "end_milepost",
        "fetched_at_utc", "source_response_url", "source_metadata_url",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"speed records fetched: {len(rows):,}")
    print(f"metadata segment keys: {len(by_segment):,}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
