#!/usr/bin/env python3
"""Extract directed link traversal speeds from MATSim events XML(.gz)."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import BinaryIO, Iterator

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **_: object):
        return iterable if iterable is not None else _NullProgress()


class _NullProgress:
    def __enter__(self):
        return self
    def __exit__(self, *_: object) -> None:
        return None
    def update(self, _: int) -> None:
        return None


class ProgressReader:
    def __init__(self, file_obj: BinaryIO, progress: object):
        self.file_obj = file_obj
        self.progress = progress
    def read(self, size: int = -1) -> bytes:
        data = self.file_obj.read(size)
        if data:
            self.progress.update(len(data))
        return data
    def close(self) -> None:
        self.file_obj.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--network-links", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def parse_float(value: str | None, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def event_stream(path: Path) -> Iterator[tuple[str, ET.Element]]:
    total = path.stat().st_size
    with path.open("rb") as raw_file:
        with tqdm(total=total, unit="B", unit_scale=True, desc=f"reading {path.name}") as progress:
            progress_reader = ProgressReader(raw_file, progress)
            if path.suffix == ".gz":
                input_file = gzip.GzipFile(fileobj=progress_reader, mode="rb")
            else:
                input_file = progress_reader
            try:
                yield from ET.iterparse(input_file, events=("end",))
            finally:
                input_file.close()


def read_link_lengths(path: Path) -> dict[str, float]:
    lengths: dict[str, float] = {}
    with path.open(newline="") as input_file:
        for row in csv.DictReader(input_file):
            link_id = row.get("link_id", "")
            length_m = parse_float(row.get("length_m"), default=0.0)
            if link_id and length_m > 0:
                lengths[f"{link_id}_fwd"] = length_m
                lengths[f"{link_id}_rev"] = length_m
    return lengths


def agent_id(element: ET.Element) -> str:
    return element.attrib.get("vehicle") or element.attrib.get("person") or element.attrib.get("driver") or ""


def main() -> int:
    args = parse_args()
    lengths = read_link_lengths(args.network_links)
    active: dict[str, tuple[str, float]] = {}
    stats: dict[str, dict[str, float]] = defaultdict(lambda: {"traversals": 0.0, "sum_distance_m": 0.0, "sum_travel_time_s": 0.0})
    events_seen = 0
    matched_traversals = 0
    unmatched_left_events = 0

    for _, element in event_stream(args.events):
        if element.tag != "event":
            element.clear()
            continue
        events_seen += 1
        event_type = element.attrib.get("type", "")
        link_id = element.attrib.get("link", "")
        who = agent_id(element)
        time_s = parse_float(element.attrib.get("time"), default=math.nan)
        if who and link_id and not math.isnan(time_s):
            if event_type == "entered link":
                active[who] = (link_id, time_s)
            elif event_type == "left link":
                entered = active.pop(who, None)
                if entered is None or entered[0] != link_id:
                    unmatched_left_events += 1
                else:
                    travel_time_s = max(0.0, time_s - entered[1])
                    length_m = lengths.get(link_id, 0.0)
                    if travel_time_s > 0 and length_m > 0:
                        row = stats[link_id]
                        row["traversals"] += 1.0
                        row["sum_distance_m"] += length_m
                        row["sum_travel_time_s"] += travel_time_s
                        matched_traversals += 1
        element.clear()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "link_id", "traversals", "sum_distance_m", "sum_travel_time_s",
        "mean_travel_time_s", "mean_speed_mps", "mean_speed_mph",
    ]
    with args.out.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for link_id in sorted(stats):
            row = stats[link_id]
            traversals = row["traversals"]
            mean_time = row["sum_travel_time_s"] / traversals if traversals else math.nan
            speed_mps = row["sum_distance_m"] / row["sum_travel_time_s"] if row["sum_travel_time_s"] else math.nan
            writer.writerow(
                {
                    "link_id": link_id,
                    "traversals": f"{traversals:.0f}",
                    "sum_distance_m": f"{row['sum_distance_m']:.8f}",
                    "sum_travel_time_s": f"{row['sum_travel_time_s']:.8f}",
                    "mean_travel_time_s": f"{mean_time:.8f}",
                    "mean_speed_mps": f"{speed_mps:.8f}",
                    "mean_speed_mph": f"{speed_mps * 2.2369362920544:.8f}",
                }
            )

    print(f"events read: {events_seen:,}")
    print(f"link traversals with durations: {matched_traversals:,}")
    print(f"unmatched left-link events: {unmatched_left_events:,}")
    print(f"links with speeds: {len(stats):,}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
