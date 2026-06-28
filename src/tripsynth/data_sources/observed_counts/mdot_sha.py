"""MDOT SHA Annual Average Daily Traffic observed-count adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from tripsynth.config import ensure_standard_directories, flatten_study_counties, read_json, write_json
from tripsynth.data_sources.observed_counts.arcgis import ArcGISFeatureServerClient
from tripsynth.data_sources.observed_counts.base import (
    ObservedCountsError,
    filter_to_counties,
    normalize_observed_counts,
    observed_missingness,
    write_geodata_outputs,
)
from tripsynth.visualization.maps import (
    write_observed_counts_coverage_html,
    write_observed_counts_coverage_png,
    write_observed_counts_sample,
)


MD_COUNTY_NAME_TO_FIPS = {
    "Allegany": "24001",
    "Anne Arundel": "24003",
    "Baltimore": "24005",
    "Calvert": "24009",
    "Caroline": "24011",
    "Carroll": "24013",
    "Cecil": "24015",
    "Charles": "24017",
    "Dorchester": "24019",
    "Frederick": "24021",
    "Garrett": "24023",
    "Harford": "24025",
    "Howard": "24027",
    "Kent": "24029",
    "Montgomery": "24031",
    "Prince George's": "24033",
    "Queen Anne's": "24035",
    "St. Mary's": "24037",
    "Somerset": "24039",
    "Talbot": "24041",
    "Washington": "24043",
    "Wicomico": "24045",
    "Worcester": "24047",
    "Baltimore City": "24510",
}


@dataclass(frozen=True)
class MDOTFetchResult:
    filtered: gpd.GeoDataFrame
    unfiltered: gpd.GeoDataFrame
    manifest: dict[str, Any]
    output_paths: dict[str, str | None]
    used_cache: bool = False


def _bbox(gdf: gpd.GeoDataFrame) -> list[float] | None:
    return list(map(float, gdf.total_bounds)) if len(gdf) else None


def _temporal_range(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    if "year" in gdf and gdf["year"].notna().any():
        return {
            "year_min": int(gdf["year"].min()),
            "year_max": int(gdf["year"].max()),
            "years": sorted(int(year) for year in gdf["year"].dropna().unique()),
        }
    return {}


def _write_mdot_manifest(manifest: dict[str, Any]) -> None:
    path = Path("data/metadata/observed_counts_mdot_sha_manifest.json")
    write_json(path, manifest)
    pd.DataFrame([manifest]).to_csv(
        "reports/tables/observed_counts_mdot_sha_manifest.csv", index=False
    )


def _write_reports(filtered: gpd.GeoDataFrame, unfiltered: gpd.GeoDataFrame) -> dict[str, str | None]:
    output_paths: dict[str, str | None] = {}
    sample_path = "reports/tables/observed_counts_mdot_sha_sample.csv"
    write_observed_counts_sample(filtered, sample_path)
    output_paths["sample_csv"] = sample_path

    missingness = observed_missingness(filtered)
    missingness["records_before_study_area_filter"] = int(len(unfiltered))
    missingness["records_after_study_area_filter"] = int(len(filtered))
    write_json("data/metadata/observed_counts_mdot_sha_missingness.json", missingness)
    pd.DataFrame([missingness]).to_csv(
        "reports/tables/observed_counts_mdot_sha_missingness.csv", index=False
    )
    output_paths["missingness_csv"] = "reports/tables/observed_counts_mdot_sha_missingness.csv"

    temporal = pd.DataFrame(
        [
            {
                "source": "mdot_sha_aadt",
                "source_temporal_type": "annual_average",
                "years_covered": sorted(int(year) for year in filtered["year"].dropna().unique())
                if "year" in filtered and filtered["year"].notna().any()
                else [],
                "date_or_month_coverage": "none",
                "overlap_with_survey_temporal_coverage": "not_evaluated_for_aadt_proxy",
                "note": "MDOT SHA AADT is annual average daily traffic and is used as spatial proxy validation.",
            }
        ]
    )
    temporal.to_csv("reports/tables/observed_counts_mdot_sha_temporal_coverage.csv", index=False)
    output_paths["temporal_report_csv"] = "reports/tables/observed_counts_mdot_sha_temporal_coverage.csv"

    html_path = "reports/maps/observed_counts_mdot_sha_coverage.html"
    write_observed_counts_coverage_html(filtered, html_path, title="MDOT SHA AADT Coverage")
    output_paths["coverage_html"] = html_path
    output_paths["coverage_png"] = write_observed_counts_coverage_png(
        filtered, "reports/maps/observed_counts_mdot_sha_coverage.png"
    )
    return output_paths


def _cached_result(processed_dir: Path) -> MDOTFetchResult | None:
    filtered_path = processed_dir / "observed_counts_mdot_sha_study_area.geoparquet"
    unfiltered_path = processed_dir / "observed_counts_mdot_sha_unfiltered.geoparquet"
    manifest_path = Path("data/metadata/observed_counts_mdot_sha_manifest.json")
    if not (filtered_path.exists() and unfiltered_path.exists() and manifest_path.exists()):
        return None
    return MDOTFetchResult(
        filtered=gpd.read_parquet(filtered_path),
        unfiltered=gpd.read_parquet(unfiltered_path),
        manifest=read_json(manifest_path),
        output_paths={
            "filtered_geoparquet": str(filtered_path),
            "unfiltered_geoparquet": str(unfiltered_path),
            "manifest_json": str(manifest_path),
            "manifest_csv": "reports/tables/observed_counts_mdot_sha_manifest.csv",
            "sample_csv": "reports/tables/observed_counts_mdot_sha_sample.csv",
            "coverage_html": "reports/maps/observed_counts_mdot_sha_coverage.html",
            "coverage_png": "reports/maps/observed_counts_mdot_sha_coverage.png",
        },
        used_cache=True,
    )


def fetch_mdot_sha_aadt(
    config: dict[str, Any],
    *,
    force: bool = False,
    client: ArcGISFeatureServerClient | None = None,
) -> MDOTFetchResult:
    ensure_standard_directories()
    source_config = config.get("observed_counts", {}).get("sources", {}).get("mdot_sha_aadt", {})
    if not source_config.get("enabled", True):
        raise ObservedCountsError("MDOT SHA AADT source is disabled in the config.")

    processed_dir = Path("data/processed/observed_counts/mdot_sha_aadt")
    if not force:
        cached = _cached_result(processed_dir)
        if cached:
            return cached

    url = source_config.get(
        "url",
        "https://mdgeodata.md.gov/imap/rest/services/Transportation/MD_AnnualAverageDailyTraffic/FeatureServer",
    )
    layer = int(source_config.get("layer", 1))
    raw_dir = Path("data/raw/observed_counts/mdot_sha_aadt")
    client = client or ArcGISFeatureServerClient(force=force)
    raw, service_meta, layer_meta = client.download_layer(
        url,
        layer=layer,
        cache_dir=raw_dir,
        result_record_count=int(source_config.get("result_record_count", 1000)),
    )
    if raw.empty:
        raise ObservedCountsError(f"No MDOT SHA AADT features downloaded from {url} layer {layer}.")

    raw = raw.copy()
    county_field = "COUNTY_DESC" if "COUNTY_DESC" in raw else "county_desc"
    if county_field in raw:
        raw["county_fips"] = raw[county_field].map(MD_COUNTY_NAME_TO_FIPS)

    field_map = dict(source_config.get("field_map", {}))
    field_map["county_fips"] = "county_fips"
    normalized = normalize_observed_counts(
        raw,
        source="mdot_sha_aadt",
        state="md",
        field_map=field_map,
        temporal_type=source_config.get("temporal_type", "annual_average"),
        default_year=int(source_config.get("default_year", 2018)),
    )

    study_counties = flatten_study_counties(config)
    md_study_counties = {county for county in study_counties if county.startswith("24")}
    filtered = filter_to_counties(normalized.gdf, md_study_counties)

    unfiltered_outputs = write_geodata_outputs(
        normalized.gdf, processed_dir, "observed_counts_mdot_sha_unfiltered"
    )
    filtered_outputs = write_geodata_outputs(
        filtered, processed_dir, "observed_counts_mdot_sha_study_area"
    )
    reports = _write_reports(filtered, normalized.gdf)

    manifest = {
        "source_name": "mdot_sha_aadt",
        "source_url": url,
        "layer": layer,
        "download_timestamp": datetime.now(timezone.utc).isoformat(),
        "query_parameters": {
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": True,
            "f_preference": "geojson",
            "study_area_counties": sorted(md_study_counties),
        },
        "raw_file_path": str(raw_dir),
        "processed_file_path": filtered_outputs.get("geoparquet"),
        "unfiltered_processed_file_path": unfiltered_outputs.get("geoparquet"),
        "number_of_records": int(len(filtered)),
        "records_before_study_area_filter": int(len(normalized.gdf)),
        "records_after_study_area_filter": int(len(filtered)),
        "crs": str(filtered.crs),
        "bbox": _bbox(filtered),
        "fields_found": normalized.fields_found,
        "fields_mapped": normalized.fields_mapped,
        "missing_fields": normalized.missing_fields,
        "temporal_fields_found": normalized.temporal_fields_found,
        "temporal_range_found": _temporal_range(filtered),
        "temporal_filtering_applied": False,
        "temporal_type": source_config.get("temporal_type", "annual_average"),
        "license_or_terms_url": service_meta.get("copyrightText")
        or service_meta.get("documentInfo", {}).get("TermsOfUse")
        or url,
        "max_record_count": layer_meta.get("maxRecordCount"),
        "gpkg_path": filtered_outputs.get("gpkg"),
        "gpkg_error": filtered_outputs.get("gpkg_error"),
    }
    _write_mdot_manifest(manifest)
    output_paths = {
        "filtered_geoparquet": filtered_outputs.get("geoparquet"),
        "filtered_gpkg": filtered_outputs.get("gpkg"),
        "unfiltered_geoparquet": unfiltered_outputs.get("geoparquet"),
        "unfiltered_gpkg": unfiltered_outputs.get("gpkg"),
        "manifest_json": "data/metadata/observed_counts_mdot_sha_manifest.json",
        "manifest_csv": "reports/tables/observed_counts_mdot_sha_manifest.csv",
        **reports,
    }
    return MDOTFetchResult(filtered, normalized.gdf, manifest, output_paths)


def format_mdot_fetch_summary(result: MDOTFetchResult) -> str:
    filtered = result.filtered
    missing_aadt = int(filtered["observed_aadt"].isna().sum()) if "observed_aadt" in filtered else 0
    retained_by_county = (
        filtered.groupby("county_fips").size().astype(int).to_dict() if len(filtered) else {}
    )
    return "\n".join(
        [
            "MDOT SHA AADT fetch complete" + (" (cached)" if result.used_cache else ""),
            f"  records retained after Maryland study-area filtering: {len(filtered)} {retained_by_county}",
            f"  AADT missingness in retained records: {missing_aadt}",
            "  source temporal type: annual_average",
            "  validation label: Maryland-only spatial roadway-volume proxy validation against MDOT SHA AADT",
            f"  outputs: {result.output_paths}",
        ]
    )
