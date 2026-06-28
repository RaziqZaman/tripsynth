"""Common observed-count schema and IO helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd


OBSERVED_COUNT_COLUMNS = [
    "source",
    "state",
    "year",
    "segment_id",
    "route_id",
    "route_name",
    "observed_aadt",
    "observed_truck_aadt_single_unit",
    "observed_truck_aadt_combination",
    "functional_class",
    "county_fips",
    "through_lanes",
    "speed_limit",
    "temporal_type",
    "count_date",
    "count_month",
    "count_time_period",
    "geometry",
]

STATE_FIPS = {"dc": "11", "md": "24", "va": "51"}


class ObservedCountsError(RuntimeError):
    """Raised when observed-count data cannot be mapped to the common schema."""


@dataclass(frozen=True)
class NormalizedCounts:
    gdf: gpd.GeoDataFrame
    fields_found: list[str]
    fields_mapped: dict[str, str]
    missing_fields: list[str]
    temporal_fields_found: list[str]


def _resolve_field(field: str | None, columns: list[str]) -> str | None:
    if not field:
        return None
    if field in columns:
        return field
    lookup = {column.lower(): column for column in columns}
    return lookup.get(str(field).lower())


def resolve_field_map(
    field_map: dict[str, str | None], columns: list[str]
) -> tuple[dict[str, str], list[str]]:
    mapped: dict[str, str] = {}
    missing: list[str] = []
    for canonical, source in field_map.items():
        resolved = _resolve_field(source, columns)
        if resolved:
            mapped[canonical] = resolved
        elif source:
            missing.append(canonical)
    return mapped, missing


def _county_fips_from_partial(state: str, values: pd.Series) -> pd.Series:
    state_fips = STATE_FIPS.get(state.lower(), "")
    text = values.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    return text.where(text.isna(), state_fips + text.str.zfill(3))


def normalize_observed_counts(
    gdf: gpd.GeoDataFrame,
    *,
    source: str,
    state: str,
    field_map: dict[str, str | None],
    temporal_type: str = "unknown",
    default_year: int | None = None,
) -> NormalizedCounts:
    columns = list(gdf.columns)
    mapped, missing = resolve_field_map(field_map, columns)
    if "observed_aadt" not in mapped:
        raise ObservedCountsError(
            "Observed count field is missing. Configure observed_counts.sources.*."
            "field_map.observed_aadt to the AADT/count field found in the source."
        )

    out = gpd.GeoDataFrame(index=gdf.index, geometry=gdf.geometry, crs=gdf.crs)
    out["source"] = source
    out["state"] = state.lower()

    if "year" in mapped:
        out["year"] = pd.to_numeric(gdf[mapped["year"]], errors="coerce").astype("Int64")
    else:
        out["year"] = default_year

    if "segment_id" in mapped:
        out["segment_id"] = gdf[mapped["segment_id"]].astype("string")
    else:
        out["segment_id"] = [f"{state.lower()}_{i}" for i in range(len(gdf))]

    optional_fields = {
        "route_id": "route_id",
        "route_name": "route_name",
        "observed_truck_aadt_single_unit": "observed_truck_aadt_single_unit",
        "observed_truck_aadt_combination": "observed_truck_aadt_combination",
        "functional_class": "functional_class",
        "through_lanes": "through_lanes",
        "speed_limit": "speed_limit",
        "count_date": "count_date",
        "count_month": "count_month",
        "count_time_period": "count_time_period",
    }
    for canonical in optional_fields:
        if canonical in mapped:
            out[canonical] = gdf[mapped[canonical]]
        else:
            out[canonical] = pd.NA

    out["observed_aadt"] = pd.to_numeric(gdf[mapped["observed_aadt"]], errors="coerce")
    for field in ["observed_truck_aadt_single_unit", "observed_truck_aadt_combination"]:
        out[field] = pd.to_numeric(out[field], errors="coerce")
    for field in ["through_lanes", "speed_limit"]:
        out[field] = pd.to_numeric(out[field], errors="coerce")

    if "county_fips" in mapped:
        out["county_fips"] = (
            gdf[mapped["county_fips"]]
            .astype("string")
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(5)
        )
    elif "county_fips_partial" in mapped:
        out["county_fips"] = _county_fips_from_partial(state, gdf[mapped["county_fips_partial"]])
    else:
        out["county_fips"] = pd.NA

    out["temporal_type"] = temporal_type
    out = out[OBSERVED_COUNT_COLUMNS]
    temporal_fields = [
        field for field in ["year", "count_date", "count_month", "count_time_period"] if field in mapped
    ]
    return NormalizedCounts(
        gdf=out,
        fields_found=columns,
        fields_mapped=mapped,
        missing_fields=missing,
        temporal_fields_found=temporal_fields,
    )


def filter_to_counties(gdf: gpd.GeoDataFrame, counties: set[str]) -> gpd.GeoDataFrame:
    if not counties or "county_fips" not in gdf:
        return gdf.copy()
    return gdf.loc[gdf["county_fips"].astype("string").isin(counties)].copy()


def write_geodata_outputs(
    gdf: gpd.GeoDataFrame, directory: Path, stem: str
) -> dict[str, str | None]:
    directory.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str | None] = {}

    parquet_path = directory / f"{stem}.geoparquet"
    gdf.to_parquet(parquet_path, index=False)
    outputs["geoparquet"] = str(parquet_path)

    gpkg_path = directory / f"{stem}.gpkg"
    try:
        gdf.to_file(gpkg_path, layer=stem[:60], driver="GPKG")
        outputs["gpkg"] = str(gpkg_path)
    except Exception as exc:  # pragma: no cover - depends on optional GDAL stack
        outputs["gpkg"] = None
        outputs["gpkg_error"] = str(exc)
    return outputs


def observed_missingness(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    geometry_valid = gdf.geometry.is_valid if len(gdf) else pd.Series(dtype=bool)
    return {
        "records": int(len(gdf)),
        "missing_aadt": int(gdf["observed_aadt"].isna().sum()) if "observed_aadt" in gdf else None,
        "records_by_state": gdf.groupby("state").size().astype(int).to_dict()
        if "state" in gdf and len(gdf)
        else {},
        "records_by_county": gdf.groupby("county_fips").size().astype(int).to_dict()
        if "county_fips" in gdf and len(gdf)
        else {},
        "crs": str(gdf.crs),
        "geometry_valid": int(geometry_valid.sum()) if len(gdf) else 0,
        "geometry_invalid": int((~geometry_valid).sum()) if len(gdf) else 0,
        "bounds": list(map(float, gdf.total_bounds)) if len(gdf) else None,
    }
