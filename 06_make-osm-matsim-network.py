#!/usr/bin/env python3
"""Convert a Geofabrik OSM PBF into a routable MATSim car network."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO
from xml.sax.saxutils import escape

import osmium
from tqdm import tqdm


INPUT_PBF = Path("06x_stage1_osm/maryland-latest.osm.pbf")
OUTPUT_NETWORK = Path("06x_stage1_osm/network.xml.gz")
OUTPUT_LINKS = Path("06x_stage1_osm/network_links.csv")
OUTPUT_LINK_MAP = Path("06x_stage1_osm/matsim_link_count_map.csv")
OUTPUT_STATS = Path("06x_stage1_osm/network_stats.json")
EARTH_RADIUS_M = 6_371_000.0

DRIVEABLE_HIGHWAYS = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
    "road",
}
BLOCKED_ACCESS_VALUES = {"no", "private", "agricultural", "forestry"}


@dataclass(frozen=True)
class WayRecord:
    way_id: int
    node_ids: tuple[int, ...]
    highway: str
    lanes: float
    freespeed: float
    capacity: float


@dataclass(frozen=True)
class SegmentRecord:
    base_id: str
    way_id: int
    sequence: int
    from_node: int
    to_node: int
    from_lon: float
    from_lat: float
    to_lon: float
    to_lat: float
    length_m: float
    highway: str
    lanes: float
    freespeed: float
    capacity: float


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}
        self.size: dict[int, int] = {}

    def add(self, item: int) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        self.add(left)
        self.add(right)
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]

    def largest_root(self) -> int | None:
        if not self.parent:
            return None
        root_sizes: dict[int, int] = {}
        for item in self.parent:
            root = self.find(item)
            root_sizes[root] = root_sizes.get(root, 0) + 1
        return max(root_sizes, key=root_sizes.get)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pbf", type=Path, default=INPUT_PBF)
    parser.add_argument("--out", type=Path, default=OUTPUT_NETWORK)
    parser.add_argument("--links-out", type=Path, default=OUTPUT_LINKS)
    parser.add_argument("--link-map-out", type=Path, default=OUTPUT_LINK_MAP)
    parser.add_argument("--stats-out", type=Path, default=OUTPUT_STATS)
    parser.add_argument(
        "--keep-disconnected",
        action="store_true",
        help="keep all driveable components instead of only the largest component",
    )
    parser.add_argument(
        "--min-segment-length-m",
        type=float,
        default=1.0,
        help="drop degenerate OSM segments shorter than this",
    )
    return parser.parse_args()


def parse_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value not in {None, ""} else default
    except ValueError:
        return default


def parse_lanes(value: str | None) -> float:
    if not value:
        return 1.0
    candidates = value.replace(";", "|").split("|")
    values = [parse_float(candidate.strip(), 0.0) for candidate in candidates]
    values = [value for value in values if value > 0]
    return max(values) if values else 1.0


def parse_maxspeed_mps(value: str | None) -> float | None:
    if not value:
        return None
    first_value = value.lower().replace(";", "|").split("|")[0].strip()
    if not first_value or first_value in {"signals", "none", "walk"}:
        return None
    parts = first_value.split()
    speed = parse_float(parts[0], 0.0)
    if speed <= 0:
        return None
    if "mph" in first_value:
        return speed * 0.44704
    return speed / 3.6


def highway_defaults(highway: str, lanes: float, maxspeed: str | None) -> tuple[float, float]:
    default_speeds = {
        "motorway": 29.06,
        "motorway_link": 22.35,
        "trunk": 26.82,
        "trunk_link": 20.12,
        "primary": 22.35,
        "primary_link": 17.88,
        "secondary": 17.88,
        "secondary_link": 15.65,
        "tertiary": 15.65,
        "tertiary_link": 13.41,
        "unclassified": 11.18,
        "residential": 11.18,
        "living_street": 5.36,
        "service": 8.94,
        "road": 11.18,
    }
    capacity_per_lane = {
        "motorway": 2000.0,
        "motorway_link": 1600.0,
        "trunk": 1800.0,
        "trunk_link": 1500.0,
        "primary": 1600.0,
        "primary_link": 1300.0,
        "secondary": 1200.0,
        "secondary_link": 1000.0,
        "tertiary": 1000.0,
        "tertiary_link": 900.0,
        "unclassified": 800.0,
        "residential": 700.0,
        "living_street": 300.0,
        "service": 500.0,
        "road": 700.0,
    }
    freespeed = parse_maxspeed_mps(maxspeed) or default_speeds.get(highway, 11.18)
    lanes = max(lanes, 1.0)
    return freespeed, capacity_per_lane.get(highway, 700.0) * lanes


def is_driveable(tags: dict[str, str]) -> bool:
    highway = tags.get("highway", "")
    if highway not in DRIVEABLE_HIGHWAYS:
        return False
    if tags.get("area") == "yes":
        return False
    for key in ["access", "vehicle", "motor_vehicle", "motorcar"]:
        if tags.get(key) in BLOCKED_ACCESS_VALUES:
            return False
    return True


def haversine_m(left_lon: float, left_lat: float, right_lon: float, right_lat: float) -> float:
    left_lat_rad = math.radians(left_lat)
    right_lat_rad = math.radians(right_lat)
    delta_lat = right_lat_rad - left_lat_rad
    delta_lon = math.radians(right_lon - left_lon)
    a = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(left_lat_rad)
        * math.cos(right_lat_rad)
        * math.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


class WayCollector(osmium.SimpleHandler):
    def __init__(self) -> None:
        super().__init__()
        self.ways: list[WayRecord] = []
        self.node_ids: set[int] = set()
        self.progress = tqdm(desc="scan OSM ways", unit="ways")

    def way(self, way: osmium.osm.Way) -> None:
        self.progress.update(1)
        tags = {tag.k: tag.v for tag in way.tags}
        if not is_driveable(tags):
            return
        node_ids = tuple(int(node.ref) for node in way.nodes)
        if len(node_ids) < 2:
            return
        lanes = parse_lanes(tags.get("lanes"))
        freespeed, capacity = highway_defaults(tags["highway"], lanes, tags.get("maxspeed"))
        self.ways.append(
            WayRecord(
                way_id=int(way.id),
                node_ids=node_ids,
                highway=tags["highway"],
                lanes=lanes,
                freespeed=freespeed,
                capacity=capacity,
            )
        )
        self.node_ids.update(node_ids)

    def close(self) -> None:
        self.progress.close()


class NodeCollector(osmium.SimpleHandler):
    def __init__(self, wanted_node_ids: set[int]) -> None:
        super().__init__()
        self.wanted_node_ids = wanted_node_ids
        self.nodes: dict[int, tuple[float, float]] = {}
        self.progress = tqdm(desc="scan OSM nodes", unit="nodes")

    def node(self, node: osmium.osm.Node) -> None:
        self.progress.update(1)
        node_id = int(node.id)
        if node_id not in self.wanted_node_ids:
            return
        if not node.location.valid():
            return
        self.nodes[node_id] = (float(node.location.lon), float(node.location.lat))

    def close(self) -> None:
        self.progress.close()


def collect_ways(path: Path) -> WayCollector:
    collector = WayCollector()
    collector.apply_file(str(path), locations=False)
    collector.close()
    return collector


def collect_nodes(path: Path, node_ids: set[int]) -> dict[int, tuple[float, float]]:
    collector = NodeCollector(node_ids)
    collector.apply_file(str(path), locations=False)
    collector.close()
    return collector.nodes


def build_segments(
    ways: list[WayRecord],
    nodes: dict[int, tuple[float, float]],
    min_segment_length_m: float,
) -> tuple[list[SegmentRecord], UnionFind]:
    segments: list[SegmentRecord] = []
    components = UnionFind()
    for way in tqdm(ways, desc="build OSM segments", unit="ways"):
        sequence = 0
        for from_node, to_node in zip(way.node_ids, way.node_ids[1:]):
            if from_node == to_node:
                continue
            from_coord = nodes.get(from_node)
            to_coord = nodes.get(to_node)
            if from_coord is None or to_coord is None:
                continue
            length_m = haversine_m(from_coord[0], from_coord[1], to_coord[0], to_coord[1])
            if length_m < min_segment_length_m:
                continue
            sequence += 1
            components.union(from_node, to_node)
            segments.append(
                SegmentRecord(
                    base_id=f"osm_{way.way_id}_{sequence}",
                    way_id=way.way_id,
                    sequence=sequence,
                    from_node=from_node,
                    to_node=to_node,
                    from_lon=from_coord[0],
                    from_lat=from_coord[1],
                    to_lon=to_coord[0],
                    to_lat=to_coord[1],
                    length_m=length_m,
                    highway=way.highway,
                    lanes=way.lanes,
                    freespeed=way.freespeed,
                    capacity=way.capacity,
                )
            )
    return segments, components


def filter_largest_component(
    segments: list[SegmentRecord],
    components: UnionFind,
    keep_disconnected: bool,
) -> list[SegmentRecord]:
    if keep_disconnected:
        return segments
    largest_root = components.largest_root()
    if largest_root is None:
        return []
    return [
        segment
        for segment in segments
        if components.find(segment.from_node) == largest_root
        and components.find(segment.to_node) == largest_root
    ]


def open_output(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def write_network(path: Path, segments: list[SegmentRecord]) -> None:
    node_coords: dict[int, tuple[float, float]] = {}
    for segment in segments:
        node_coords[segment.from_node] = (segment.from_lon, segment.from_lat)
        node_coords[segment.to_node] = (segment.to_lon, segment.to_lat)

    with open_output(path) as output_file:
        output_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        output_file.write('<!DOCTYPE network SYSTEM "http://www.matsim.org/files/dtd/network_v2.dtd">\n')
        output_file.write('<network name="maryland_osm_stage1">\n')
        output_file.write('  <nodes>\n')
        for node_id, (lon, lat) in tqdm(
            sorted(node_coords.items()),
            desc="write MATSim nodes",
            unit="nodes",
        ):
            output_file.write(f'    <node id="n{node_id}" x="{lon:.8f}" y="{lat:.8f}" />\n')
        output_file.write('  </nodes>\n')
        output_file.write('  <links capperiod="01:00:00" effectivecellsize="7.5" effectivelanewidth="3.75">\n')
        for segment in tqdm(segments, desc="write MATSim links", unit="segments"):
            for suffix, from_node, to_node in [
                ("fwd", segment.from_node, segment.to_node),
                ("rev", segment.to_node, segment.from_node),
            ]:
                output_file.write(
                    f'    <link id="{escape(segment.base_id + "_" + suffix)}" '
                    f'from="n{from_node}" to="n{to_node}" length="{segment.length_m:.3f}" '
                    f'freespeed="{segment.freespeed:.3f}" capacity="{segment.capacity:.3f}" '
                    f'permlanes="{segment.lanes:.3f}" modes="car" />\n'
                )
        output_file.write('  </links>\n')
        output_file.write('</network>\n')


def write_links(path: Path, segments: list[SegmentRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output_file:
        fieldnames = [
            "link_id",
            "way_id",
            "sequence",
            "from_lon",
            "from_lat",
            "to_lon",
            "to_lat",
            "length_m",
            "highway",
            "lanes",
            "freespeed",
            "capacity",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for segment in tqdm(segments, desc="write network_links.csv", unit="segments"):
            writer.writerow(
                {
                    "link_id": segment.base_id,
                    "way_id": segment.way_id,
                    "sequence": segment.sequence,
                    "from_lon": f"{segment.from_lon:.8f}",
                    "from_lat": f"{segment.from_lat:.8f}",
                    "to_lon": f"{segment.to_lon:.8f}",
                    "to_lat": f"{segment.to_lat:.8f}",
                    "length_m": f"{segment.length_m:.3f}",
                    "highway": segment.highway,
                    "lanes": f"{segment.lanes:.3f}",
                    "freespeed": f"{segment.freespeed:.3f}",
                    "capacity": f"{segment.capacity:.3f}",
                }
            )


def write_link_map(path: Path, segments: list[SegmentRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["matsim_link_id", "count_link_id", "direction"],
        )
        writer.writeheader()
        for segment in tqdm(segments, desc="write link map", unit="segments"):
            writer.writerow(
                {
                    "matsim_link_id": f"{segment.base_id}_fwd",
                    "count_link_id": segment.base_id,
                    "direction": "fwd",
                }
            )
            writer.writerow(
                {
                    "matsim_link_id": f"{segment.base_id}_rev",
                    "count_link_id": segment.base_id,
                    "direction": "rev",
                }
            )


def write_stats(
    path: Path,
    ways: list[WayRecord],
    node_count: int,
    raw_segments: list[SegmentRecord],
    kept_segments: list[SegmentRecord],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    highway_counts: dict[str, int] = {}
    for segment in kept_segments:
        highway_counts[segment.highway] = highway_counts.get(segment.highway, 0) + 1
    stats = {
        "ways": len(ways),
        "candidate_nodes": node_count,
        "raw_segments": len(raw_segments),
        "kept_segments": len(kept_segments),
        "matsim_directed_links": len(kept_segments) * 2,
        "highway_segment_counts": dict(sorted(highway_counts.items())),
    }
    path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    if not args.pbf.exists():
        raise FileNotFoundError(
            f"{args.pbf} does not exist; run 06_download-osm-network.py first"
        )

    ways = collect_ways(args.pbf)
    nodes = collect_nodes(args.pbf, ways.node_ids)
    raw_segments, components = build_segments(ways.ways, nodes, args.min_segment_length_m)
    kept_segments = filter_largest_component(
        raw_segments,
        components,
        args.keep_disconnected,
    )

    write_network(args.out, kept_segments)
    write_links(args.links_out, kept_segments)
    write_link_map(args.link_map_out, kept_segments)
    write_stats(args.stats_out, ways.ways, len(nodes), raw_segments, kept_segments)

    print(f"wrote {args.out}")
    print(f"wrote {args.links_out}")
    print(f"wrote {args.link_map_out}")
    print(f"wrote {args.stats_out}")
    print(f"driveable ways: {len(ways.ways):,}")
    print(f"osm nodes used: {len(nodes):,}")
    print(f"segments kept: {len(kept_segments):,} / {len(raw_segments):,}")
    print(f"matsim directed links: {len(kept_segments) * 2:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
