"""Shortest-path routing on the MDOT SHA AADT line network."""

from __future__ import annotations

from collections import defaultdict
from heapq import heappop, heappush
from itertools import count
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Point

from tripsynth.preprocessing.tract_geometries import (
    load_or_fetch_tract_centroids,
    normalize_tract_fips,
)
from tripsynth.routing.base import RoutingResult
from tripsynth.routing.desire_lines import _county_centroids, tract_to_county_fips


NodeKey = tuple[int, int]


def _iter_lines(geometry):
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        yield from geometry.geoms


def _node_key(x: float, y: float, snap_tolerance_m: float) -> NodeKey:
    return (int(round(x / snap_tolerance_m)), int(round(y / snap_tolerance_m)))


def _build_mdot_graph(
    observed_counts: gpd.GeoDataFrame,
    projected_crs: str,
    *,
    snap_tolerance_m: float,
) -> tuple[nx.MultiGraph, dict[str, set[NodeKey]]]:
    graph = nx.MultiGraph()
    county_nodes: dict[str, set[NodeKey]] = defaultdict(set)
    observed = observed_counts.to_crs(projected_crs)

    for row in observed.itertuples(index=False):
        segment_id = str(row.segment_id)
        county_fips = str(row.county_fips) if pd.notna(row.county_fips) else None
        for line in _iter_lines(row.geometry):
            coords = list(line.coords)
            if len(coords) < 2:
                continue
            for start, end in zip(coords[:-1], coords[1:]):
                sx, sy = float(start[0]), float(start[1])
                ex, ey = float(end[0]), float(end[1])
                if sx == ex and sy == ey:
                    continue
                u = _node_key(sx, sy, snap_tolerance_m)
                v = _node_key(ex, ey, snap_tolerance_m)
                if u == v:
                    continue
                ux, uy = u[0] * snap_tolerance_m, u[1] * snap_tolerance_m
                vx, vy = v[0] * snap_tolerance_m, v[1] * snap_tolerance_m
                graph.add_node(u, x=ux, y=uy)
                graph.add_node(v, x=vx, y=vy)
                if county_fips:
                    county_nodes[county_fips].update([u, v])
                geometry = LineString([(ux, uy), (vx, vy)])
                graph.add_edge(
                    u,
                    v,
                    length_m=float(geometry.length),
                    observed_segment_id=segment_id,
                    county_fips=county_fips,
                    geometry=geometry,
                )
    return graph, county_nodes


def _nearest_node_index(graph: nx.MultiGraph) -> tuple[list[NodeKey], cKDTree]:
    nodes = list(graph.nodes)
    coords = np.array([[graph.nodes[node]["x"], graph.nodes[node]["y"]] for node in nodes])
    return nodes, cKDTree(coords)


def _nearest_node(point: Point, nodes: list[NodeKey], tree: cKDTree) -> NodeKey:
    _, idx = tree.query([point.x, point.y], k=1)
    return nodes[int(idx)]


def _nearby_distinct_nodes(
    point: Point,
    origin: NodeKey,
    nodes: list[NodeKey],
    tree: cKDTree,
    graph: nx.MultiGraph,
    *,
    max_candidates: int = 75,
    min_distance_m: float = 100.0,
) -> list[NodeKey]:
    if not nodes:
        return []
    k = min(len(nodes), max(2, max_candidates))
    _, idxs = tree.query([point.x, point.y], k=k)
    idx_array = np.atleast_1d(idxs)
    ox, oy = graph.nodes[origin]["x"], graph.nodes[origin]["y"]
    close_any_distance = []
    separated = []
    for idx in idx_array:
        node = nodes[int(idx)]
        if node == origin:
            continue
        dx = graph.nodes[node]["x"] - ox
        dy = graph.nodes[node]["y"] - oy
        distance = float((dx * dx + dy * dy) ** 0.5)
        close_any_distance.append(node)
        if distance >= min_distance_m:
            separated.append(node)
    return separated or close_any_distance


def _farthest_county_node(
    origin: NodeKey,
    candidates: set[NodeKey],
    graph: nx.MultiGraph,
) -> NodeKey | None:
    if not candidates:
        return None
    ox, oy = graph.nodes[origin]["x"], graph.nodes[origin]["y"]
    best_node = None
    best_distance = -1.0
    for node in candidates:
        dx = graph.nodes[node]["x"] - ox
        dy = graph.nodes[node]["y"] - oy
        distance = dx * dx + dy * dy
        if distance > best_distance:
            best_node = node
            best_distance = distance
    return best_node


def _edge_data_for_step(graph: nx.MultiGraph, u, v) -> dict[str, Any]:
    edges = graph.get_edge_data(u, v)
    return min(edges.values(), key=lambda data: data.get("length_m", float("inf")))


def _edge_weight(graph: nx.MultiGraph, u: NodeKey, v: NodeKey) -> float:
    edges = graph.get_edge_data(u, v)
    return float(min(data.get("length_m", 1.0) for data in edges.values()))


def _multi_target_shortest_paths(
    graph: nx.MultiGraph,
    source: NodeKey,
    targets: list[NodeKey],
) -> tuple[dict[NodeKey, float], dict[NodeKey, list[NodeKey]]]:
    target_set = {target for target in targets if target != source}
    if not target_set:
        return {}, {}

    push_count = count()
    fringe: list[tuple[float, int, NodeKey]] = [(0.0, next(push_count), source)]
    settled: dict[NodeKey, float] = {}
    seen: dict[NodeKey, float] = {source: 0.0}
    predecessor: dict[NodeKey, NodeKey] = {}
    found: set[NodeKey] = set()

    while fringe and found != target_set:
        distance, _, node = heappop(fringe)
        if node in settled:
            continue
        settled[node] = distance
        if node in target_set:
            found.add(node)
            if found == target_set:
                break
        for neighbor, keyed_edges in graph[node].items():
            if neighbor in settled:
                continue
            step_length = float(min(data.get("length_m", 1.0) for data in keyed_edges.values()))
            next_distance = distance + step_length
            if next_distance < seen.get(neighbor, float("inf")):
                seen[neighbor] = next_distance
                predecessor[neighbor] = node
                heappush(fringe, (next_distance, next(push_count), neighbor))

    paths: dict[NodeKey, list[NodeKey]] = {}
    distances: dict[NodeKey, float] = {}
    for target in found:
        reverse_path = [target]
        while reverse_path[-1] != source:
            previous = predecessor.get(reverse_path[-1])
            if previous is None:
                reverse_path = []
                break
            reverse_path.append(previous)
        if reverse_path:
            paths[target] = list(reversed(reverse_path))
            distances[target] = settled[target]
    return distances, paths


def _routing_config(config: dict[str, Any]) -> dict[str, Any]:
    return (
        config.get("routing", {}).get("mdot_shortest_path")
        or config.get("validation", {}).get("routing", {}).get("mdot_shortest_path", {})
    )


def _group_od_volumes(
    trips: pd.DataFrame,
    group_cols: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    working = trips.copy()
    volume_col = config.get("validation", {}).get("baseline", {}).get("route_volume_col")
    if volume_col and volume_col in working:
        working["_route_volume"] = pd.to_numeric(working[volume_col], errors="coerce").fillna(0.0)
    else:
        working["_route_volume"] = 1.0
    return (
        working.groupby(group_cols, dropna=False)
        .agg(predicted_volume=("_route_volume", "sum"), synthetic_trips=("_route_volume", "size"))
        .reset_index()
    )


def _accumulate_path(
    graph: nx.MultiGraph,
    path: list[NodeKey],
    predicted_volume: float,
    volume_by_segment: dict[str, float],
) -> tuple[float, int]:
    path_length = 0.0
    touched_segments: set[str] = set()
    for u, v in zip(path[:-1], path[1:]):
        data = _edge_data_for_step(graph, u, v)
        segment_id = str(data["observed_segment_id"])
        volume_by_segment[segment_id] += float(predicted_volume)
        path_length += float(data.get("length_m", 0.0))
        touched_segments.add(segment_id)
    return path_length, len(touched_segments)


def _observed_with_predicted(
    observed_counts: gpd.GeoDataFrame,
    volume_by_segment: dict[str, float],
) -> gpd.GeoDataFrame:
    observed = observed_counts.copy()
    observed["observed_segment_id"] = observed["segment_id"].astype("string")
    observed["predicted_volume"] = observed["observed_segment_id"].map(volume_by_segment).fillna(0.0)
    return observed.loc[observed["predicted_volume"] > 0].copy()


def _empty_result(observed_counts: gpd.GeoDataFrame, metadata: dict[str, Any]) -> RoutingResult:
    return RoutingResult(
        method="mdot_shortest_path",
        routed_volumes=gpd.GeoDataFrame(geometry=[], crs=observed_counts.crs),
        metadata=metadata,
    )


def _route_county_shortest_paths(
    trips: pd.DataFrame,
    observed_counts: gpd.GeoDataFrame,
    config: dict[str, Any],
    graph: nx.MultiGraph,
    county_nodes: dict[str, set[NodeKey]],
    nodes: list[NodeKey],
    tree: cKDTree,
    projected_crs: str,
    snap_tolerance_m: float,
) -> RoutingResult:
    centers = _county_centroids(observed_counts, projected_crs)
    center_by_county = dict(zip(centers["county_fips"], centers.geometry))

    working = trips.copy()
    working["origin_county_fips"] = working["origin_tract"].map(tract_to_county_fips)
    working["destination_county_fips"] = working["destination_tract"].map(tract_to_county_fips)
    working = working.dropna(subset=["origin_county_fips", "destination_county_fips"])
    working = working[
        working["origin_county_fips"].isin(center_by_county)
        & working["destination_county_fips"].isin(center_by_county)
    ].copy()
    grouped = _group_od_volumes(working, ["origin_county_fips", "destination_county_fips"], config)

    volume_by_segment: dict[str, float] = defaultdict(float)
    route_records: list[dict[str, Any]] = []
    failures = 0
    same_county_routes = 0
    no_path = 0

    for row in grouped.itertuples(index=False):
        origin_county = str(row.origin_county_fips)
        destination_county = str(row.destination_county_fips)
        origin_node = _nearest_node(center_by_county[origin_county], nodes, tree)
        if origin_county == destination_county:
            same_county_routes += 1
            destination_node = _farthest_county_node(origin_node, county_nodes[origin_county], graph)
        else:
            destination_node = _nearest_node(center_by_county[destination_county], nodes, tree)
        if destination_node is None or destination_node == origin_node:
            failures += 1
            continue
        try:
            path = nx.shortest_path(graph, origin_node, destination_node, weight="length_m")
        except nx.NetworkXNoPath:
            failures += 1
            no_path += 1
            continue

        path_length, touched_count = _accumulate_path(
            graph, path, float(row.predicted_volume), volume_by_segment
        )
        route_records.append(
            {
                "route_id": f"{origin_county}_{destination_county}",
                "origin_county_fips": origin_county,
                "destination_county_fips": destination_county,
                "predicted_volume": float(row.predicted_volume),
                "synthetic_trips": int(row.synthetic_trips),
                "path_length_m": path_length,
                "segments_touched": touched_count,
            }
        )

    routed = _observed_with_predicted(observed_counts, volume_by_segment)
    metadata = {
        "method": "mdot_shortest_path",
        "od_geography": "county",
        "input_trips": int(len(trips)),
        "vehicle_trips_with_supported_counties": int(len(working)),
        "od_pairs": int(len(grouped)),
        "routes": int(len(route_records)),
        "same_county_routes": int(same_county_routes),
        "failed_routes": int(failures),
        "no_path_routes": int(no_path),
        "graph_nodes": int(graph.number_of_nodes()),
        "graph_edges": int(graph.number_of_edges()),
        "routed_observed_segments": int(len(routed)),
        "projected_crs": projected_crs,
        "snap_tolerance_m": snap_tolerance_m,
        "caveat": (
            "County-level OD volumes are routed over the MDOT AADT line graph. "
            "Use only as fallback when tract centroids are unavailable."
        ),
    }
    return RoutingResult(method="mdot_shortest_path", routed_volumes=routed, metadata=metadata)


def _route_tract_shortest_paths(
    trips: pd.DataFrame,
    observed_counts: gpd.GeoDataFrame,
    config: dict[str, Any],
    graph: nx.MultiGraph,
    nodes: list[NodeKey],
    tree: cKDTree,
    projected_crs: str,
    snap_tolerance_m: float,
    routing_config: dict[str, Any],
) -> RoutingResult:
    allow_fetch = bool(routing_config.get("auto_fetch_tracts", False))
    tract_centers = load_or_fetch_tract_centroids(config, projected_crs, allow_fetch=allow_fetch)
    supported_counties = set(observed_counts["county_fips"].dropna().astype("string"))
    if supported_counties:
        tract_centers = tract_centers.loc[
            tract_centers["county_fips"].astype("string").isin(supported_counties)
        ].copy()
    center_by_tract = dict(zip(tract_centers["tract_fips"], tract_centers.geometry))
    county_by_tract = dict(zip(tract_centers["tract_fips"], tract_centers["county_fips"].astype("string")))

    working = trips.copy()
    working["origin_tract_fips"] = working["origin_tract"].map(normalize_tract_fips)
    working["destination_tract_fips"] = working["destination_tract"].map(normalize_tract_fips)
    working = working.dropna(subset=["origin_tract_fips", "destination_tract_fips"])
    working = working[
        working["origin_tract_fips"].isin(center_by_tract)
        & working["destination_tract_fips"].isin(center_by_tract)
    ].copy()
    grouped = _group_od_volumes(working, ["origin_tract_fips", "destination_tract_fips"], config)

    max_od_pairs = routing_config.get("max_od_pairs")
    if max_od_pairs is not None:
        grouped = grouped.sort_values("predicted_volume", ascending=False).head(int(max_od_pairs)).copy()

    node_by_tract = {
        tract: _nearest_node(point, nodes, tree) for tract, point in center_by_tract.items()
    }
    volume_by_segment: dict[str, float] = defaultdict(float)
    route_records: list[dict[str, Any]] = []
    failures = 0
    same_tract_routes = 0
    same_node_routes = 0
    no_path = 0
    route_jobs_by_origin: dict[NodeKey, list[dict[str, Any]]] = defaultdict(list)

    for row in grouped.itertuples(index=False):
        origin_tract = str(row.origin_tract_fips)
        destination_tract = str(row.destination_tract_fips)
        origin_node = node_by_tract[origin_tract]
        destination_node = node_by_tract[destination_tract]
        candidate_nodes = [destination_node]

        if origin_tract == destination_tract:
            same_tract_routes += 1
        if destination_node == origin_node:
            same_node_routes += 1
            candidate_nodes = _nearby_distinct_nodes(
                center_by_tract[destination_tract],
                origin_node,
                nodes,
                tree,
                graph,
                max_candidates=int(routing_config.get("same_node_candidates", 75)),
                min_distance_m=float(routing_config.get("same_node_min_distance_m", 100)),
            )

        candidate_nodes = [candidate for candidate in candidate_nodes if candidate is not None and candidate != origin_node]
        if not candidate_nodes:
            failures += 1
            no_path += 1
            continue
        route_jobs_by_origin[origin_node].append(
            {
                "origin_tract": origin_tract,
                "destination_tract": destination_tract,
                "candidate_nodes": candidate_nodes,
                "predicted_volume": float(row.predicted_volume),
                "synthetic_trips": int(row.synthetic_trips),
            }
        )

    for origin_node, jobs in route_jobs_by_origin.items():
        targets: list[NodeKey] = []
        for job in jobs:
            targets.extend(job["candidate_nodes"])
        distances, paths = _multi_target_shortest_paths(graph, origin_node, targets)

        for job in jobs:
            available = [
                (distances[candidate], candidate, paths[candidate])
                for candidate in job["candidate_nodes"]
                if candidate in paths
            ]
            if not available:
                failures += 1
                no_path += 1
                continue
            _, destination_node, path = min(available, key=lambda item: item[0])
            origin_tract = job["origin_tract"]
            destination_tract = job["destination_tract"]
            path_length, touched_count = _accumulate_path(
                graph, path, float(job["predicted_volume"]), volume_by_segment
            )
            route_records.append(
                {
                    "route_id": f"{origin_tract}_{destination_tract}",
                    "origin_tract_fips": origin_tract,
                    "destination_tract_fips": destination_tract,
                    "origin_county_fips": str(county_by_tract.get(origin_tract, origin_tract[:5])),
                    "destination_county_fips": str(county_by_tract.get(destination_tract, destination_tract[:5])),
                    "predicted_volume": float(job["predicted_volume"]),
                    "synthetic_trips": int(job["synthetic_trips"]),
                    "path_length_m": path_length,
                    "segments_touched": touched_count,
                    "origin_node": str(origin_node),
                    "destination_node": str(destination_node),
                }
            )

    routed = _observed_with_predicted(observed_counts, volume_by_segment)
    metadata = {
        "method": "mdot_shortest_path",
        "od_geography": "tract",
        "input_trips": int(len(trips)),
        "vehicle_trips_with_supported_tracts": int(len(working)),
        "tract_centroids_available": int(len(center_by_tract)),
        "supported_counties": sorted(supported_counties),
        "od_pairs": int(len(grouped)),
        "routes": int(len(route_records)),
        "same_tract_routes": int(same_tract_routes),
        "same_node_routes": int(same_node_routes),
        "failed_routes": int(failures),
        "no_path_routes": int(no_path),
        "graph_nodes": int(graph.number_of_nodes()),
        "graph_edges": int(graph.number_of_edges()),
        "routed_observed_segments": int(len(routed)),
        "projected_crs": projected_crs,
        "snap_tolerance_m": snap_tolerance_m,
        "caveat": (
            "Tract OD volumes are routed over the MDOT SHA AADT line graph using tract centroids. "
            "Validation is restricted to tracts in counties covered by the Maryland MDOT AADT source."
        ),
    }
    return RoutingResult(method="mdot_shortest_path", routed_volumes=routed, metadata=metadata)


def route_mdot_shortest_paths(
    trips: pd.DataFrame,
    observed_counts: gpd.GeoDataFrame,
    config: dict[str, Any],
) -> RoutingResult:
    """Route OD volumes on the MDOT AADT segment graph."""
    projected_crs = config.get("project", {}).get("crs_projected", "EPSG:26918")
    routing_config = _routing_config(config)
    snap_tolerance_m = float(routing_config.get("snap_tolerance_m", 15))
    graph, county_nodes = _build_mdot_graph(
        observed_counts, projected_crs, snap_tolerance_m=snap_tolerance_m
    )
    if graph.number_of_edges() == 0:
        return _empty_result(
            observed_counts,
            {
                "method": "mdot_shortest_path",
                "routes": 0,
                "error": "MDOT graph has no edges.",
            },
        )

    nodes, tree = _nearest_node_index(graph)
    od_geography = str(routing_config.get("od_geography", "county")).lower()
    if od_geography == "tract":
        try:
            return _route_tract_shortest_paths(
                trips,
                observed_counts,
                config,
                graph,
                nodes,
                tree,
                projected_crs,
                snap_tolerance_m,
                routing_config,
            )
        except (FileNotFoundError, ValueError) as exc:
            if not bool(routing_config.get("fallback_to_county_centroids", True)):
                raise
            routed = _route_county_shortest_paths(
                trips,
                observed_counts,
                config,
                graph,
                county_nodes,
                nodes,
                tree,
                projected_crs,
                snap_tolerance_m,
            )
            routed.metadata["requested_od_geography"] = "tract"
            routed.metadata["fallback_reason"] = str(exc)
            return routed

    if od_geography != "county":
        raise ValueError("mdot_shortest_path.od_geography must be 'tract' or 'county'.")
    return _route_county_shortest_paths(
        trips,
        observed_counts,
        config,
        graph,
        county_nodes,
        nodes,
        tree,
        projected_crs,
        snap_tolerance_m,
    )
