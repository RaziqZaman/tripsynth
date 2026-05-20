#!/usr/bin/env python3
"""Prepare county-level population-scale MATSim scenarios."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import TextIO
from xml.sax.saxutils import escape

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **_: object):
        return iterable if iterable is not None else _NullProgress()


class _NullProgress:
    def __enter__(self): return self
    def __exit__(self, *_: object) -> None: return None
    def update(self, _: int) -> None: return None


MD_FIPS_TO_COUNTY = {
    "24001": "Allegany",
    "24003": "Anne Arundel",
    "24005": "Baltimore",
    "24009": "Calvert",
    "24011": "Caroline",
    "24013": "Carroll",
    "24015": "Cecil",
    "24017": "Charles",
    "24019": "Dorchester",
    "24021": "Frederick",
    "24023": "Garrett",
    "24025": "Harford",
    "24027": "Howard",
    "24029": "Kent",
    "24031": "Montgomery",
    "24033": "Prince George's",
    "24035": "Queen Anne's",
    "24037": "St. Mary's",
    "24039": "Somerset",
    "24041": "Talbot",
    "24043": "Washington",
    "24045": "Wicomico",
    "24047": "Worcester",
    "24510": "Baltimore City",
}

REAL_CSV = Path("03x_filled-survey.csv")
SYNTHETIC_CSV = Path("07x_population_scale/synthetic_population_trips.csv")
TAZ_CENTROIDS = Path("06x_stage1/taz_centroids.csv")
NETWORK_LINKS = Path("06x_stage1_osm/network_links.csv")
OBSERVED_COUNTS = Path("06x_stage1_osm/observed_counts.csv")
OUT_ROOT = Path("08x_county_runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-csv", type=Path, default=REAL_CSV)
    parser.add_argument("--synthetic-csv", type=Path, default=SYNTHETIC_CSV)
    parser.add_argument("--taz-centroids", type=Path, default=TAZ_CENTROIDS)
    parser.add_argument("--network-links", type=Path, default=NETWORK_LINKS)
    parser.add_argument("--observed-counts", type=Path, default=OBSERVED_COUNTS)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--counties", default="", help="comma-separated county names; blank means all Maryland counties with observed counts")
    parser.add_argument("--bbox-buffer-deg", type=float, default=0.15)
    parser.add_argument("--demand-mode", choices=["both", "either"], default="both", help="both keeps county-internal OD trips; either keeps trips touching county")
    parser.add_argument("--min-observed-counts", type=int, default=10)
    parser.add_argument("--min-vehicle-trips", type=float, default=100.0)
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parse_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value not in {None, ""} else default
    except ValueError:
        return default


def open_text(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def read_taz_centroids(path: Path) -> tuple[dict[str, str], dict[str, tuple[float, float]], dict[str, list[dict[str, str]]]]:
    taz_to_county: dict[str, str] = {}
    taz_xy: dict[str, tuple[float, float]] = {}
    county_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            taz = row.get("taz", "")
            county = MD_FIPS_TO_COUNTY.get(row.get("fipsstco", ""), "")
            if not taz or not county:
                continue
            taz_to_county[taz] = county
            lon = parse_float(row.get("lon") or row.get("x"))
            lat = parse_float(row.get("lat") or row.get("y"))
            taz_xy[taz] = (lon, lat)
            county_rows[county].append(row)
    return taz_to_county, taz_xy, county_rows


def read_observed_by_county(path: Path) -> dict[str, list[dict[str, str]]]:
    by_county: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="") as input_file:
        for row in csv.DictReader(input_file):
            county = row.get("county", "")
            if county:
                by_county[county].append(row)
    return by_county


def selected_counties(raw: str, observed_by_county: dict[str, list[dict[str, str]]], min_observed: int) -> list[str]:
    if raw.strip():
        requested = [item.strip() for item in raw.split(",") if item.strip()]
        observed_names = {name.lower(): name for name in observed_by_county}
        result = []
        for county in requested:
            result.append(observed_names.get(county.lower(), county))
        return result
    return sorted(county for county, rows in observed_by_county.items() if len(rows) >= min_observed)


def bbox_for_county(
    county: str,
    county_taz_rows: list[dict[str, str]],
    observed_rows: list[dict[str, str]],
    buffer_deg: float,
) -> tuple[float, float, float, float]:
    lons: list[float] = []
    lats: list[float] = []
    for row in county_taz_rows:
        lons.append(parse_float(row.get("lon") or row.get("x")))
        lats.append(parse_float(row.get("lat") or row.get("y")))
    for row in observed_rows:
        lons.append(parse_float(row.get("lon")))
        lats.append(parse_float(row.get("lat")))
    if not lons or not lats:
        raise ValueError(f"No coordinates for county {county}")
    return min(lons) - buffer_deg, max(lons) + buffer_deg, min(lats) - buffer_deg, max(lats) + buffer_deg


def link_midpoint(row: dict[str, str]) -> tuple[float, float]:
    lon = (parse_float(row["from_lon"]) + parse_float(row["to_lon"])) / 2.0
    lat = (parse_float(row["from_lat"]) + parse_float(row["to_lat"])) / 2.0
    return lon, lat


def write_network_and_map(
    network_links: Path,
    out_network: Path,
    out_links: Path,
    out_map: Path,
    bbox: tuple[float, float, float, float],
) -> int:
    min_lon, max_lon, min_lat, max_lat = bbox
    nodes: dict[tuple[str, str], str] = {}
    kept_rows: list[dict[str, str]] = []

    def node_id(lon: str, lat: str) -> str:
        key = (f"{parse_float(lon):.7f}", f"{parse_float(lat):.7f}")
        found = nodes.get(key)
        if found is None:
            found = f"n{len(nodes) + 1}"
            nodes[key] = found
        return found

    with network_links.open(newline="") as input_file, out_links.open("w", newline="") as links_file:
        reader = csv.DictReader(input_file)
        writer = csv.DictWriter(links_file, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in tqdm(reader, desc=f"filter network {out_network.parent.name}", unit="links"):
            lon, lat = link_midpoint(row)
            if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
                row = dict(row)
                row["from_node"] = node_id(row["from_lon"], row["from_lat"])
                row["to_node"] = node_id(row["to_lon"], row["to_lat"])
                kept_rows.append(row)
                writer.writerow({key: row.get(key, "") for key in writer.fieldnames or []})

    with open_text(out_network) as output_file:
        output_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        output_file.write('<!DOCTYPE network SYSTEM "http://www.matsim.org/files/dtd/network_v2.dtd">\n')
        output_file.write('<network>\n')
        output_file.write('  <nodes>\n')
        for (lon, lat), node in nodes.items():
            output_file.write(f'    <node id="{node}" x="{lon}" y="{lat}" />\n')
        output_file.write('  </nodes>\n')
        output_file.write('  <links capperiod="01:00:00">\n')
        for row in kept_rows:
            attrs = {
                "length": parse_float(row.get("length_m"), 1.0),
                "freespeed": parse_float(row.get("freespeed"), 11.18),
                "capacity": parse_float(row.get("capacity"), 700.0),
                "permlanes": parse_float(row.get("lanes"), 1.0),
            }
            for direction, from_node, to_node in [
                ("fwd", row["from_node"], row["to_node"]),
                ("rev", row["to_node"], row["from_node"]),
            ]:
                link_id = escape(f"{row['link_id']}_{direction}")
                output_file.write(
                    f'    <link id="{link_id}" from="{from_node}" to="{to_node}" '
                    f'length="{attrs["length"]:.3f}" freespeed="{attrs["freespeed"]:.3f}" '
                    f'capacity="{attrs["capacity"]:.3f}" permlanes="{attrs["permlanes"]:.3f}" modes="car" />\n'
                )
        output_file.write('  </links>\n')
        output_file.write('</network>\n')

    with out_map.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["matsim_link_id", "count_link_id", "direction"])
        writer.writeheader()
        for row in kept_rows:
            writer.writerow({"matsim_link_id": f"{row['link_id']}_fwd", "count_link_id": row["link_id"], "direction": "fwd"})
            writer.writerow({"matsim_link_id": f"{row['link_id']}_rev", "count_link_id": row["link_id"], "direction": "rev"})

    return len(kept_rows)


def include_for_county(row: dict[str, str], county: str, taz_to_county: dict[str, str], mode: str) -> bool:
    origin = taz_to_county.get(row.get("o_tpb_taz", ""), "")
    destination = taz_to_county.get(row.get("d_tpb_taz", ""), "")
    if mode == "either":
        return origin == county or destination == county
    return origin == county and destination == county


def vehicle_row(row: dict[str, str], scenario: str, row_id: int, person_weight: float) -> dict[str, str] | None:
    occupancy = parse_float(row.get("vehicle_occupancy"), 0.0)
    if occupancy <= 0:
        return None
    vehicle_weight = person_weight / occupancy
    return {
        "scenario": scenario,
        "source_row": str(row_id),
        "o_tpb_taz": row.get("o_tpb_taz", ""),
        "d_tpb_taz": row.get("d_tpb_taz", ""),
        "o_activity": row.get("o_activity", ""),
        "d_activity": row.get("d_activity", ""),
        "departure_time_min": row.get("departure_time_min", ""),
        "travel_mode": row.get("travel_mode", ""),
        "vehicle_occupancy": row.get("vehicle_occupancy", ""),
        "person_weight": f"{person_weight:.8f}",
        "vehicle_weight_unscaled": f"{vehicle_weight:.8f}",
        "vehicle_weight": f"{vehicle_weight:.8f}",
    }


def write_vehicle_demands(
    real_csv: Path,
    synthetic_csv: Path,
    out_root: Path,
    counties: list[str],
    taz_to_county: dict[str, str],
    mode: str,
) -> dict[str, dict[str, float]]:
    fieldnames = [
        "scenario", "source_row", "o_tpb_taz", "d_tpb_taz", "o_activity", "d_activity",
        "departure_time_min", "travel_mode", "vehicle_occupancy", "person_weight",
        "vehicle_weight_unscaled", "vehicle_weight",
    ]
    county_set = set(counties)
    handles: dict[tuple[str, str], TextIO] = {}
    writers: dict[tuple[str, str], csv.DictWriter] = {}
    totals: dict[str, dict[str, float]] = {county: defaultdict(float) for county in counties}

    def writer_for(county: str, scenario: str) -> csv.DictWriter:
        key = (county, scenario)
        if key not in writers:
            path = out_root / slugify(county) / f"{scenario}_vehicle_trips.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            handles[key] = path.open("w", newline="")
            writers[key] = csv.DictWriter(handles[key], fieldnames=fieldnames)
            writers[key].writeheader()
        return writers[key]

    try:
        with real_csv.open(newline="") as input_file:
            for row_id, row in enumerate(tqdm(csv.DictReader(input_file), desc="real demand", unit="rows"), start=1):
                county = taz_to_county.get(row.get("o_tpb_taz", ""), "")
                if county not in county_set or not include_for_county(row, county, taz_to_county, mode):
                    continue
                out = vehicle_row(row, "real", row_id, parse_float(row.get("wthhfin"), 0.0))
                if out is None:
                    continue
                writer_for(county, "real").writerow(out)
                totals[county]["real_rows"] += 1
                totals[county]["real_person_trips"] += parse_float(out["person_weight"])
                totals[county]["real_vehicle_trips"] += parse_float(out["vehicle_weight"])

        with synthetic_csv.open(newline="") as input_file:
            for row_id, row in enumerate(tqdm(csv.DictReader(input_file), desc="synthetic demand", unit="rows"), start=1):
                county = taz_to_county.get(row.get("o_tpb_taz", ""), "")
                if county not in county_set or not include_for_county(row, county, taz_to_county, mode):
                    continue
                out = vehicle_row(row, "synthetic", row_id, 1.0)
                if out is None:
                    continue
                writer_for(county, "synthetic").writerow(out)
                totals[county]["synthetic_rows"] += 1
                totals[county]["synthetic_person_trips"] += 1.0
                totals[county]["synthetic_vehicle_trips"] += parse_float(out["vehicle_weight"])
    finally:
        for handle in handles.values():
            handle.close()

    for county in counties:
        county_dir = out_root / slugify(county)
        county_dir.mkdir(parents=True, exist_ok=True)
        for scenario in ["real", "synthetic"]:
            path = county_dir / f"{scenario}_vehicle_trips.csv"
            if not path.exists():
                with path.open("w", newline="") as output_file:
                    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                    writer.writeheader()
        with (county_dir / "scenario_totals.csv").open("w", newline="") as output_file:
            fieldnames_totals = ["scenario", "input_rows", "included_rows", "person_trips_unscaled", "vehicle_trips_unscaled", "scale", "vehicle_trips_scaled", "road_modes"]
            writer = csv.DictWriter(output_file, fieldnames=fieldnames_totals)
            writer.writeheader()
            writer.writerow({
                "scenario": "real", "input_rows": "", "included_rows": f"{totals[county]['real_rows']:.0f}",
                "person_trips_unscaled": f"{totals[county]['real_person_trips']:.8f}",
                "vehicle_trips_unscaled": f"{totals[county]['real_vehicle_trips']:.8f}",
                "scale": "1.000000000000", "vehicle_trips_scaled": f"{totals[county]['real_vehicle_trips']:.8f}",
                "road_modes": "positive_occupancy",
            })
            writer.writerow({
                "scenario": "synthetic", "input_rows": "", "included_rows": f"{totals[county]['synthetic_rows']:.0f}",
                "person_trips_unscaled": f"{totals[county]['synthetic_person_trips']:.8f}",
                "vehicle_trips_unscaled": f"{totals[county]['synthetic_vehicle_trips']:.8f}",
                "scale": "1.000000000000", "vehicle_trips_scaled": f"{totals[county]['synthetic_vehicle_trips']:.8f}",
                "road_modes": "positive_occupancy",
            })
    return totals


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    taz_to_county, _, county_taz_rows = read_taz_centroids(args.taz_centroids)
    observed_by_county = read_observed_by_county(args.observed_counts)
    counties = selected_counties(args.counties, observed_by_county, args.min_observed_counts)

    manifest_rows = []
    for county in counties:
        county_dir = args.out_root / slugify(county)
        county_dir.mkdir(parents=True, exist_ok=True)
        observed_rows = observed_by_county.get(county, [])
        with (county_dir / "observed_counts.csv").open("w", newline="") as output_file:
            fieldnames = ["link_id", "observed_volume", "count_id", "match_distance_m", "lon", "lat", "road_name", "route_id", "county"]
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(observed_rows)
        bbox = bbox_for_county(county, county_taz_rows.get(county, []), observed_rows, args.bbox_buffer_deg)
        network_links = write_network_and_map(
            args.network_links,
            county_dir / "network.xml.gz",
            county_dir / "network_links.csv",
            county_dir / "matsim_link_count_map.csv",
            bbox,
        )
        manifest_rows.append({
            "county": county,
            "slug": slugify(county),
            "county_dir": str(county_dir),
            "observed_counts": len(observed_rows),
            "network_base_links": network_links,
            "bbox_min_lon": bbox[0], "bbox_max_lon": bbox[1], "bbox_min_lat": bbox[2], "bbox_max_lat": bbox[3],
        })

    totals = write_vehicle_demands(args.real_csv, args.synthetic_csv, args.out_root, counties, taz_to_county, args.demand_mode)
    runnable_rows = []
    for row in manifest_rows:
        county = row["county"]
        row.update({
            "real_vehicle_trips": f"{totals[county]['real_vehicle_trips']:.8f}",
            "synthetic_vehicle_trips": f"{totals[county]['synthetic_vehicle_trips']:.8f}",
            "real_rows": f"{totals[county]['real_rows']:.0f}",
            "synthetic_rows": f"{totals[county]['synthetic_rows']:.0f}",
        })
        row["runnable"] = (
            int(row["observed_counts"]) >= args.min_observed_counts
            and float(row["real_vehicle_trips"]) >= args.min_vehicle_trips
            and float(row["synthetic_vehicle_trips"]) >= args.min_vehicle_trips
            and int(row["network_base_links"]) > 0
        )
        if row["runnable"]:
            runnable_rows.append(row)

    manifest_path = args.out_root / "county_manifest.csv"
    with manifest_path.open("w", newline="") as output_file:
        fieldnames = list(manifest_rows[0]) if manifest_rows else []
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    runnable_rows.sort(key=lambda row: max(float(row["real_vehicle_trips"]), float(row["synthetic_vehicle_trips"])))
    with (args.out_root / "county_run_list.txt").open("w") as output_file:
        for row in runnable_rows:
            output_file.write(f"{row['slug']}\n")
    print(f"wrote {manifest_path}")
    print(f"runnable counties: {len(runnable_rows)} / {len(manifest_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
