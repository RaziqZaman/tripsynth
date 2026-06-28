"""Desire-line routing proxy for the first runnable baseline."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from tripsynth.routing.base import RoutingResult


def tract_to_county_fips(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().replace(".0", "")
    if not text or text.lower() == "nan":
        return None
    try:
        text = str(int(float(text)))
    except ValueError:
        text = "".join(ch for ch in text if ch.isdigit())
    if len(text) < 5:
        return None
    return text.zfill(11)[:5]


def _county_centroids(observed_counts: gpd.GeoDataFrame, projected_crs: str) -> gpd.GeoDataFrame:
    if observed_counts.empty:
        return gpd.GeoDataFrame(columns=["county_fips", "geometry"], geometry="geometry", crs=projected_crs)
    observed = observed_counts.dropna(subset=["county_fips"]).to_crs(projected_crs)
    rows = []
    for county, sub in observed.groupby("county_fips"):
        rows.append({"county_fips": str(county), "geometry": sub.geometry.union_all().centroid})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=projected_crs)


def route_county_desire_lines(
    trips: pd.DataFrame,
    observed_counts: gpd.GeoDataFrame,
    config: dict[str, Any],
) -> RoutingResult:
    """Aggregate vehicle trips to county OD desire lines.

    This is intentionally a spatial proxy baseline. It does not claim roadway path
    realism; it provides a reproducible first route-like volume surface for AADT
    spatial validation while shortest-path/network assignment is being added.
    """
    projected_crs = config.get("project", {}).get("crs_projected", "EPSG:26918")
    baseline_config = config.get("validation", {}).get("baseline", {})
    volume_col = baseline_config.get("route_volume_col")
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

    if volume_col and volume_col in working:
        working["_route_volume"] = pd.to_numeric(working[volume_col], errors="coerce").fillna(0)
    else:
        working["_route_volume"] = 1.0

    grouped = (
        working.groupby(["origin_county_fips", "destination_county_fips"], dropna=False)
        .agg(predicted_volume=("_route_volume", "sum"), synthetic_trips=("_route_volume", "size"))
        .reset_index()
    )

    records = []
    for row in grouped.itertuples(index=False):
        origin = center_by_county[row.origin_county_fips]
        destination = center_by_county[row.destination_county_fips]
        if origin.equals(destination):
            geometry = LineString([(origin.x - 500, origin.y), (origin.x + 500, origin.y)])
        else:
            geometry = LineString([origin, destination])
        route_id = f"{row.origin_county_fips}_{row.destination_county_fips}"
        records.append(
            {
                "route_id": route_id,
                "origin_county_fips": row.origin_county_fips,
                "destination_county_fips": row.destination_county_fips,
                "predicted_volume": float(row.predicted_volume),
                "synthetic_trips": int(row.synthetic_trips),
                "geometry": geometry,
            }
        )

    routed = gpd.GeoDataFrame(records, geometry="geometry", crs=projected_crs)
    metadata = {
        "method": "tract_desire_line_proxy",
        "input_trips": int(len(trips)),
        "vehicle_trips_with_supported_counties": int(len(working)),
        "routes": int(len(routed)),
        "projected_crs": projected_crs,
        "caveat": "County-centroid desire lines are a routing proxy, not network assignment.",
    }
    return RoutingResult(method="tract_desire_line_proxy", routed_volumes=routed, metadata=metadata)
