#!/usr/bin/env python3
"""Extract directed link volumes from a MATSim events XML(.gz) file."""

from __future__ import annotations

import argparse
import csv
import gzip
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import BinaryIO, Iterator

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for bare environments.
    def tqdm(iterable=None, **_: object):
        return iterable if iterable is not None else _NullProgress()


class _NullProgress:
    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def update(self, _: int) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--event-type",
        default="entered link",
        help="MATSim event type to count; default counts link-entry events",
    )
    return parser.parse_args()


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


def event_stream(path: Path) -> Iterator[tuple[str, ET.Element]]:
    total = path.stat().st_size
    with path.open("rb") as raw_file:
        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"reading {path.name}",
        ) as progress:
            progress_reader = ProgressReader(raw_file, progress)
            if path.suffix == ".gz":
                input_file = gzip.GzipFile(fileobj=progress_reader, mode="rb")
            else:
                input_file = progress_reader
            try:
                yield from ET.iterparse(input_file, events=("end",))
            finally:
                input_file.close()


def main() -> int:
    args = parse_args()
    volumes: Counter[str] = Counter()
    events_seen = 0

    for _, element in event_stream(args.events):
            if element.tag != "event":
                element.clear()
                continue
            events_seen += 1
            if element.attrib.get("type") == args.event_type:
                link_id = element.attrib.get("link")
                if link_id:
                    volumes[link_id] += 1
            element.clear()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["link_id", "volume"])
        writer.writeheader()
        for link_id, volume in sorted(volumes.items()):
            writer.writerow({"link_id": link_id, "volume": volume})

    print(f"events read: {events_seen}")
    print(f"links counted: {len(volumes)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
