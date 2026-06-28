"""Runnable weighted-resampling + routed AADT proxy baseline experiment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tripsynth.config import ensure_standard_directories, write_json
from tripsynth.data_sources.observed_counts.loaders import load_observed_counts_for_validation
from tripsynth.data_sources.survey import build_synthesis_frame, load_survey
from tripsynth.preprocessing.survey_audit import run_survey_audit
from tripsynth.routing.desire_lines import route_county_desire_lines
from tripsynth.routing.mdot_shortest_path import route_mdot_shortest_paths
from tripsynth.synthesis.weighted_resampling import sample_by_scale
from tripsynth.validation.spatial_proxy import validate_spatial_aadt_proxy


@dataclass(frozen=True)
class BaselineRunResult:
    run_dir: Path
    summary: dict[str, Any]


def _filter_vehicle_trips(trips: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    vehicle_config = config.get("validation", {}).get("vehicle_trip_filter", {})
    mode_field = vehicle_config.get("mode_field", "mode")
    auto_modes = {str(value) for value in vehicle_config.get("auto_modes", [])}
    include_missing = bool(vehicle_config.get("include_when_mode_missing", False))

    if mode_field not in trips or trips[mode_field].isna().all():
        if include_missing:
            return trips.copy(), {
                "mode_field": mode_field,
                "auto_modes": sorted(auto_modes),
                "included_without_mode": True,
                "vehicle_trip_count": int(len(trips)),
            }
        raise ValueError(
            f"Vehicle-count validation needs a mapped mode field or include_when_mode_missing=true. "
            f"Configured mode_field={mode_field!r}."
        )

    mode_text = trips[mode_field].astype("string").str.replace(r"\.0$", "", regex=True)
    filtered = trips.loc[mode_text.isin(auto_modes)].copy()
    return filtered, {
        "mode_field": mode_field,
        "auto_modes": sorted(auto_modes),
        "input_trip_count": int(len(trips)),
        "vehicle_trip_count": int(len(filtered)),
        "excluded_trip_count": int(len(trips) - len(filtered)),
    }


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_weighted_resampling_desire_line_baseline(
    config: dict[str, Any],
    *,
    scale_factor: float | None = None,
    seed: int | None = None,
    run_dir: str | Path | None = None,
    force_counts: bool = False,
    observed_source: str | None = None,
) -> BaselineRunResult:
    ensure_standard_directories()
    audit = run_survey_audit(config)
    survey = load_survey(config)
    observed, observed_meta = load_observed_counts_for_validation(
        config, source_name=observed_source, force_fetch=force_counts
    )

    baseline_config = config.get("validation", {}).get("baseline", {})
    scale = float(scale_factor if scale_factor is not None else baseline_config.get("scale_factor", 1))
    run_seed = int(seed if seed is not None else config.get("project", {}).get("seed", 42))

    synthesis_frame = build_synthesis_frame(survey, config)
    vehicle_trips, vehicle_filter_meta = _filter_vehicle_trips(synthesis_frame, config)
    synthesis = sample_by_scale(vehicle_trips, scale_factor=scale, seed=run_seed)
    routing_method = baseline_config.get("routing_method", "mdot_shortest_path")
    if routing_method == "mdot_shortest_path":
        routed = route_mdot_shortest_paths(synthesis.synthetic_trips, observed, config)
    elif routing_method in {"tract_desire_line_proxy", "tract_border_crossing"}:
        routed = route_county_desire_lines(synthesis.synthetic_trips, observed, config)
    else:
        raise ValueError(f"Unsupported baseline routing_method {routing_method!r}")
    metrics_by_segment, metrics_by_run = validate_spatial_aadt_proxy(
        routed.routed_volumes, observed, config
    )

    run_path = Path(run_dir) if run_dir else Path("outputs/runs") / f"baseline_{_timestamp()}"
    for child in ["synthetic_trips", "routed_volumes", "metrics", "tables"]:
        (run_path / child).mkdir(parents=True, exist_ok=True)

    synthetic_path = run_path / "synthetic_trips" / "weighted_resampling.csv"
    routed_path = run_path / "routed_volumes" / f"{routed.method}.geoparquet"
    segment_path = run_path / "metrics" / "metrics_by_segment.parquet"
    metrics_path = run_path / "metrics" / "metrics_by_run.csv"
    summary_path = run_path / "tables" / "baseline_summary.json"
    route_table_path = run_path / "tables" / "route_table.csv"
    failed_routes_path = run_path / "tables" / "failed_routes.csv"

    synthesis.synthetic_trips.to_csv(synthetic_path, index=False)
    routed.routed_volumes.to_parquet(routed_path, index=False)
    metrics_by_segment.to_parquet(segment_path, index=False)
    metrics_by_run.to_csv(metrics_path, index=False)
    route_table = routed.route_table if routed.route_table is not None else pd.DataFrame()
    failed_routes = routed.failed_routes if routed.failed_routes is not None else pd.DataFrame()
    route_table.to_csv(route_table_path, index=False)
    failed_routes.to_csv(failed_routes_path, index=False)

    summary = {
        "run_dir": str(run_path),
        "validation_mode": "spatial_aadt_proxy",
        "validation_label": "spatial roadway-volume proxy validation against AADT",
        "survey_coverage": {
            "covered_years": audit.metadata.get("covered_years"),
            "covered_months_by_year": audit.metadata.get("covered_months_by_year"),
            "full_year_coverage_by_year": audit.metadata.get("full_year_coverage_by_year"),
            "first_survey_date": audit.metadata.get("first_survey_date"),
            "last_survey_date": audit.metadata.get("last_survey_date"),
        },
        "observed_counts": {
            **observed_meta,
            "records": int(len(observed)),
            "temporal_type": "annual_average",
        },
        "vehicle_filter": vehicle_filter_meta,
        "synthesis": synthesis.metadata,
        "routing": routed.metadata,
        "outputs": {
            "synthetic_trips": str(synthetic_path),
            "routed_volumes": str(routed_path),
            "metrics_by_segment": str(segment_path),
            "metrics_by_run": str(metrics_path),
            "route_table": str(route_table_path),
            "failed_routes": str(failed_routes_path),
            "summary": str(summary_path),
        },
        "interpretation_caveat": (
            "This baseline validates spatial roadway-volume patterns against AADT. "
            "When routing_method is mdot_shortest_path, tract-level OD volumes are assigned "
            "over the MDOT AADT line graph when tract geometries are available; "
            "absolute-volume metrics still require "
            "calibration/expansion assumptions."
        ),
    }
    write_json(summary_path, summary)
    return BaselineRunResult(run_dir=run_path, summary=summary)
