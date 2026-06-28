"""FHWA HPMS 2018 observed-count adapter."""

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


@dataclass(frozen=True)
class FetchCountsResult:
    filtered: gpd.GeoDataFrame
    unfiltered: gpd.GeoDataFrame
    manifests: list[dict[str, Any]]
    output_paths: dict[str, str | None]
    used_cache: bool = False


def _bbox(gdf: gpd.GeoDataFrame) -> list[float] | None:
    return list(map(float, gdf.total_bounds)) if len(gdf) else None


def _temporal_range(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "year" in gdf and gdf["year"].notna().any():
        out["year_min"] = int(gdf["year"].min())
        out["year_max"] = int(gdf["year"].max())
        out["years"] = sorted(int(year) for year in gdf["year"].dropna().unique())
    if "count_date" in gdf and gdf["count_date"].notna().any():
        dates = pd.to_datetime(gdf["count_date"], errors="coerce").dropna()
        if not dates.empty:
            out["date_min"] = dates.min().date().isoformat()
            out["date_max"] = dates.max().date().isoformat()
    if "count_month" in gdf and gdf["count_month"].notna().any():
        out["months"] = sorted(str(value) for value in gdf["count_month"].dropna().unique())
    return out


def _write_manifest_files(manifests: list[dict[str, Any]]) -> None:
    write_json("data/metadata/observed_counts_manifest.json", manifests)
    rows = []
    for manifest in manifests:
        row = manifest.copy()
        for field in [
            "query_parameters",
            "fields_found",
            "fields_mapped",
            "missing_fields",
            "temporal_fields_found",
            "temporal_range_found",
            "bbox",
        ]:
            row[field] = str(row.get(field))
        rows.append(row)
    pd.DataFrame(rows).to_csv("reports/tables/observed_counts_manifest.csv", index=False)


def _write_reports(filtered: gpd.GeoDataFrame, unfiltered: gpd.GeoDataFrame) -> dict[str, str | None]:
    output_paths: dict[str, str | None] = {}
    sample_path = "reports/tables/observed_counts_sample.csv"
    write_observed_counts_sample(filtered, sample_path)
    output_paths["sample_csv"] = sample_path

    missingness = observed_missingness(filtered)
    missingness["records_before_study_area_filter"] = int(len(unfiltered))
    missingness["records_after_study_area_filter"] = int(len(filtered))
    write_json("data/metadata/observed_counts_missingness.json", missingness)
    pd.DataFrame([missingness]).to_csv("reports/tables/observed_counts_missingness.csv", index=False)
    output_paths["missingness_csv"] = "reports/tables/observed_counts_missingness.csv"

    temporal = pd.DataFrame(
        [
            {
                "source": "fhwa_hpms_2018",
                "source_temporal_type": "annual_average",
                "years_covered": sorted(
                    int(year) for year in filtered["year"].dropna().unique()
                )
                if "year" in filtered
                else [],
                "date_or_month_coverage": "none",
                "overlap_with_survey_temporal_coverage": "not_evaluated_for_aadt_proxy",
                "note": "HPMS AADT is annual average daily traffic and is used as spatial proxy validation by default.",
            }
        ]
    )
    temporal.to_csv("reports/tables/observed_counts_temporal_coverage.csv", index=False)
    output_paths["temporal_report_csv"] = "reports/tables/observed_counts_temporal_coverage.csv"

    html_path = "reports/maps/observed_counts_coverage.html"
    write_observed_counts_coverage_html(filtered, html_path, title="FHWA HPMS 2018 DMV AADT Coverage")
    output_paths["coverage_html"] = html_path
    output_paths["coverage_png"] = write_observed_counts_coverage_png(
        filtered, "reports/maps/observed_counts_coverage.png"
    )
    return output_paths


def _cached_result(processed_dir: Path) -> FetchCountsResult | None:
    filtered_path = processed_dir / "observed_counts_hpms_2018_dmv.geoparquet"
    unfiltered_path = processed_dir / "observed_counts_hpms_2018_dmv_unfiltered.geoparquet"
    manifest_path = Path("data/metadata/observed_counts_manifest.json")
    if not (filtered_path.exists() and unfiltered_path.exists() and manifest_path.exists()):
        return None
    filtered = gpd.read_parquet(filtered_path)
    unfiltered = gpd.read_parquet(unfiltered_path)
    manifests = read_json(manifest_path)
    output_paths = {
        "filtered_geoparquet": str(filtered_path),
        "unfiltered_geoparquet": str(unfiltered_path),
        "manifest_json": str(manifest_path),
        "manifest_csv": "reports/tables/observed_counts_manifest.csv",
        "sample_csv": "reports/tables/observed_counts_sample.csv",
        "coverage_html": "reports/maps/observed_counts_coverage.html",
        "coverage_png": "reports/maps/observed_counts_coverage.png",
    }
    return FetchCountsResult(filtered, unfiltered, manifests, output_paths, used_cache=True)


def fetch_fhwa_hpms_2018(
    config: dict[str, Any],
    *,
    force: bool = False,
    client: ArcGISFeatureServerClient | None = None,
) -> FetchCountsResult:
    ensure_standard_directories()
    source_config = config.get("observed_counts", {}).get("sources", {}).get("fhwa_hpms_2018", {})
    if not source_config.get("enabled", True):
        raise ObservedCountsError("FHWA HPMS 2018 source is disabled in the config.")

    processed_dir = Path("data/processed/observed_counts/fhwa_hpms_2018")
    if not force:
        cached = _cached_result(processed_dir)
        if cached:
            return cached

    client = client or ArcGISFeatureServerClient(force=force)
    raw_root = Path("data/raw/observed_counts/fhwa_hpms_2018")
    all_frames: list[gpd.GeoDataFrame] = []
    manifests: list[dict[str, Any]] = []
    download_timestamp = datetime.now(timezone.utc).isoformat()

    for state, state_config in source_config.get("states", {}).items():
        url = state_config["url"]
        layer = int(state_config.get("layer", 0))
        raw_dir = raw_root / state
        raw_gdf, service_meta, layer_meta = client.download_layer(
            url,
            layer=layer,
            cache_dir=raw_dir,
        )
        if raw_gdf.empty:
            raise ObservedCountsError(f"No features downloaded from {url} layer {layer}.")

        normalized = normalize_observed_counts(
            raw_gdf,
            source="fhwa_hpms_2018",
            state=state,
            field_map=source_config.get("field_map", {}),
            temporal_type=source_config.get("temporal_type", "annual_average"),
            default_year=2018,
        )
        state_outputs = write_geodata_outputs(
            normalized.gdf,
            processed_dir,
            f"observed_counts_hpms_2018_{state}_unfiltered",
        )
        manifests.append(
            {
                "source_name": "fhwa_hpms_2018",
                "state": state,
                "source_url": url,
                "layer": layer,
                "download_timestamp": download_timestamp,
                "query_parameters": {
                    "where": "1=1",
                    "outFields": "*",
                    "returnGeometry": True,
                    "f_preference": "geojson",
                },
                "raw_file_path": str(raw_dir),
                "processed_file_path": state_outputs.get("geoparquet"),
                "number_of_records": int(len(normalized.gdf)),
                "crs": str(normalized.gdf.crs),
                "bbox": _bbox(normalized.gdf),
                "fields_found": normalized.fields_found,
                "fields_mapped": normalized.fields_mapped,
                "missing_fields": normalized.missing_fields,
                "temporal_fields_found": normalized.temporal_fields_found,
                "temporal_range_found": _temporal_range(normalized.gdf),
                "temporal_filtering_applied": False,
                "temporal_type": source_config.get("temporal_type", "annual_average"),
                "license_or_terms_url": service_meta.get("documentInfo", {}).get("TermsOfUse")
                or service_meta.get("copyrightText")
                or url,
                "max_record_count": layer_meta.get("maxRecordCount"),
                "gpkg_path": state_outputs.get("gpkg"),
                "gpkg_error": state_outputs.get("gpkg_error"),
            }
        )
        all_frames.append(normalized.gdf)

    if not all_frames:
        raise ObservedCountsError("No HPMS states were configured for download.")

    unfiltered = gpd.GeoDataFrame(pd.concat(all_frames, ignore_index=True), crs=all_frames[0].crs)
    study_counties = flatten_study_counties(config)
    filtered = filter_to_counties(unfiltered, study_counties)

    unfiltered_outputs = write_geodata_outputs(
        unfiltered,
        processed_dir,
        "observed_counts_hpms_2018_dmv_unfiltered",
    )
    filtered_outputs = write_geodata_outputs(
        filtered,
        processed_dir,
        "observed_counts_hpms_2018_dmv",
    )
    reports = _write_reports(filtered, unfiltered)

    manifests.append(
        {
            "source_name": "fhwa_hpms_2018_dmv_harmonized",
            "state": "dmv",
            "source_url": ";".join(
                state_config["url"] for state_config in source_config.get("states", {}).values()
            ),
            "layer": "configured per state",
            "download_timestamp": download_timestamp,
            "query_parameters": {"study_area_counties": sorted(study_counties)},
            "raw_file_path": str(raw_root),
            "processed_file_path": filtered_outputs.get("geoparquet"),
            "number_of_records": int(len(filtered)),
            "crs": str(filtered.crs),
            "bbox": _bbox(filtered),
            "fields_found": sorted(set().union(*(set(frame.columns) for frame in all_frames))),
            "fields_mapped": source_config.get("field_map", {}),
            "missing_fields": [],
            "temporal_fields_found": ["year"],
            "temporal_range_found": _temporal_range(filtered),
            "temporal_filtering_applied": False,
            "temporal_type": source_config.get("temporal_type", "annual_average"),
            "license_or_terms_url": "See source FeatureServer metadata for each state.",
            "records_before_study_area_filter": int(len(unfiltered)),
            "records_after_study_area_filter": int(len(filtered)),
            "unfiltered_processed_file_path": unfiltered_outputs.get("geoparquet"),
            "gpkg_path": filtered_outputs.get("gpkg"),
            "gpkg_error": filtered_outputs.get("gpkg_error"),
        }
    )
    _write_manifest_files(manifests)

    output_paths: dict[str, str | None] = {
        "filtered_geoparquet": filtered_outputs.get("geoparquet"),
        "filtered_gpkg": filtered_outputs.get("gpkg"),
        "unfiltered_geoparquet": unfiltered_outputs.get("geoparquet"),
        "unfiltered_gpkg": unfiltered_outputs.get("gpkg"),
        "manifest_json": "data/metadata/observed_counts_manifest.json",
        "manifest_csv": "reports/tables/observed_counts_manifest.csv",
        **reports,
    }
    return FetchCountsResult(filtered, unfiltered, manifests, output_paths)


def format_fetch_summary(result: FetchCountsResult) -> str:
    filtered = result.filtered
    unfiltered = result.unfiltered
    by_state = (
        unfiltered.groupby("state").size().astype(int).to_dict() if len(unfiltered) else {}
    )
    retained_by_state = (
        filtered.groupby("state").size().astype(int).to_dict() if len(filtered) else {}
    )
    missing_aadt = int(filtered["observed_aadt"].isna().sum()) if "observed_aadt" in filtered else 0
    return "\n".join(
        [
            "Observed counts fetch complete" + (" (cached)" if result.used_cache else ""),
            f"  records downloaded by state: {by_state}",
            f"  records retained after study-area filtering: {len(filtered)} {retained_by_state}",
            f"  AADT missingness in retained records: {missing_aadt}",
            "  source temporal type: annual_average",
            "  validation label: spatial roadway-volume proxy validation against AADT",
            f"  outputs: {result.output_paths}",
        ]
    )
