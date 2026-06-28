"""Run spatial validation across multiple synthesis methods."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tripsynth.config import ensure_standard_directories, write_json
from tripsynth.data_sources.observed_counts.loaders import load_observed_counts_for_validation
from tripsynth.data_sources.survey import build_synthesis_frame, load_survey
from tripsynth.experiments.baseline import _filter_vehicle_trips
from tripsynth.preprocessing.survey_audit import run_survey_audit
from tripsynth.reporting.baseline_diagnostics import run_baseline_diagnostics
from tripsynth.routing.desire_lines import route_county_desire_lines
from tripsynth.routing.mdot_shortest_path import route_mdot_shortest_paths
from tripsynth.synthesis.factory import synthesize_trips
from tripsynth.validation.spatial_proxy import validate_spatial_aadt_proxy


@dataclass(frozen=True)
class MethodComparisonResult:
    run_dir: Path
    comparison: pd.DataFrame
    summary: dict[str, Any]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_methods(config: dict[str, Any]) -> list[str]:
    return list(config.get("experiment", {}).get("synthesis_methods") or ["weighted_resampling"])


def _default_seeds(config: dict[str, Any]) -> list[int]:
    return [int(config.get("project", {}).get("seed", 42))]


def _metric_row(metrics_by_run: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for record in metrics_by_run.to_dict("records"):
        label = record.get("comparison_type")
        if not label:
            continue
        for key, value in record.items():
            if key == "comparison_type" or pd.isna(value):
                continue
            out[f"{label}.{key}"] = value
    return out


def _route_synthetic_trips(synthetic: pd.DataFrame, observed, config: dict[str, Any]):
    routing_method = config.get("validation", {}).get("baseline", {}).get(
        "routing_method", "mdot_shortest_path"
    )
    if routing_method == "mdot_shortest_path":
        return route_mdot_shortest_paths(synthetic, observed, config)
    if routing_method in {"tract_desire_line_proxy", "tract_border_crossing"}:
        return route_county_desire_lines(synthetic, observed, config)
    raise ValueError(f"Unsupported routing_method {routing_method!r}")


def _write_method_run(
    *,
    run_path: Path,
    method: str,
    seed: int,
    validation_label: str,
    audit,
    observed,
    observed_meta: dict[str, Any],
    vehicle_filter_meta: dict[str, Any],
    synthesis,
    routed,
    metrics_by_segment,
    metrics_by_run,
) -> dict[str, Any]:
    for child in ["synthetic_trips", "routed_volumes", "metrics", "tables"]:
        (run_path / child).mkdir(parents=True, exist_ok=True)

    synthetic_path = run_path / "synthetic_trips" / f"{method}.csv"
    routed_path = run_path / "routed_volumes" / f"{routed.method}.geoparquet"
    segment_path = run_path / "metrics" / "metrics_by_segment.parquet"
    metrics_path = run_path / "metrics" / "metrics_by_run.csv"
    route_table_path = run_path / "tables" / "route_table.csv"
    failed_routes_path = run_path / "tables" / "failed_routes.csv"
    summary_path = run_path / "tables" / "baseline_summary.json"

    synthesis.synthetic_trips.to_csv(synthetic_path, index=False)
    routed.routed_volumes.to_parquet(routed_path, index=False)
    metrics_by_segment.to_parquet(segment_path, index=False)
    metrics_by_run.to_csv(metrics_path, index=False)
    (routed.route_table if routed.route_table is not None else pd.DataFrame()).to_csv(
        route_table_path, index=False
    )
    (routed.failed_routes if routed.failed_routes is not None else pd.DataFrame()).to_csv(
        failed_routes_path, index=False
    )

    summary = {
        "run_dir": str(run_path),
        "synthesis_method": method,
        "seed": int(seed),
        "validation_mode": "spatial_aadt_proxy",
        "validation_label": validation_label,
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
            "This method comparison validates spatial roadway-volume patterns against AADT. "
            "AADT is an annual-average proxy; absolute volumes require calibration/expansion assumptions."
        ),
    }
    write_json(summary_path, summary)
    return summary


def run_synthesis_method_comparison(
    config: dict[str, Any],
    *,
    methods: list[str] | None = None,
    seeds: list[int] | None = None,
    scale_factor: float | None = None,
    run_dir: str | Path | None = None,
    force_counts: bool = False,
    observed_source: str | None = None,
    diagnostics: bool = False,
) -> MethodComparisonResult:
    ensure_standard_directories()
    audit = run_survey_audit(config)
    survey = load_survey(config)
    observed, observed_meta = load_observed_counts_for_validation(
        config, source_name=observed_source, force_fetch=force_counts
    )
    synthesis_frame = build_synthesis_frame(survey, config)
    vehicle_trips, vehicle_filter_meta = _filter_vehicle_trips(synthesis_frame, config)

    baseline_config = config.get("validation", {}).get("baseline", {})
    scale = float(scale_factor if scale_factor is not None else baseline_config.get("scale_factor", 1))
    n = max(1, int(round(len(vehicle_trips) * scale)))
    methods = methods or _default_methods(config)
    seeds = seeds or _default_seeds(config)
    validation_label = baseline_config.get(
        "validation_label", "spatial roadway-volume proxy validation against AADT"
    )
    root = Path(run_dir) if run_dir else Path("outputs/runs") / f"method_comparison_{_timestamp()}"
    root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    method_summaries = []
    for method in methods:
        for seed in seeds:
            method_run = root / f"{method}_seed_{seed}"
            synthesis = synthesize_trips(method, vehicle_trips, n=n, seed=int(seed), config=config)
            routed = _route_synthetic_trips(synthesis.synthetic_trips, observed, config)
            metrics_by_segment, metrics_by_run = validate_spatial_aadt_proxy(
                routed.routed_volumes, observed, config
            )
            summary = _write_method_run(
                run_path=method_run,
                method=method,
                seed=int(seed),
                validation_label=validation_label,
                audit=audit,
                observed=observed,
                observed_meta=observed_meta,
                vehicle_filter_meta=vehicle_filter_meta,
                synthesis=synthesis,
                routed=routed,
                metrics_by_segment=metrics_by_segment,
                metrics_by_run=metrics_by_run,
            )
            if diagnostics:
                run_baseline_diagnostics(method_run)
            row = {
                "synthesis_method": method,
                "seed": int(seed),
                "run_dir": str(method_run),
                "synthetic_trips": int(len(synthesis.synthetic_trips)),
                "supported_vehicle_trips": int(
                    routed.metadata.get("vehicle_trips_with_supported_tracts")
                    or routed.metadata.get("vehicle_trips_with_supported_counties")
                    or 0
                ),
                "od_pairs": int(routed.metadata.get("od_pairs", 0) or 0),
                "routes": int(routed.metadata.get("routes", 0) or 0),
                "failed_routes": int(routed.metadata.get("failed_routes", 0) or 0),
                "routed_observed_segments": int(routed.metadata.get("routed_observed_segments", 0) or 0),
                "synthesis_implementation": synthesis.metadata.get("implementation", method),
                **_metric_row(metrics_by_run),
            }
            rows.append(row)
            method_summaries.append(summary)

    comparison = pd.DataFrame(rows)
    comparison_path = root / "comparison_by_run.csv"
    comparison.to_csv(comparison_path, index=False)
    compact_columns = [
        column
        for column in [
            "synthesis_method",
            "seed",
            "synthetic_trips",
            "supported_vehicle_trips",
            "od_pairs",
            "routes",
            "failed_routes",
            "routed_observed_segments",
            "uncalibrated_absolute_proxy.pearson_correlation",
            "uncalibrated_absolute_proxy.spearman_correlation",
            "uncalibrated_absolute_proxy.jensen_shannon_divergence",
            "uncalibrated_absolute_proxy.top_10pct_overlap",
            "relative_share_validation.share_spearman_correlation",
            "relative_share_validation.share_jensen_shannon_divergence",
            "run_dir",
        ]
        if column in comparison.columns
    ]
    compact_path = root / "comparison_compact.csv"
    comparison[compact_columns].to_csv(compact_path, index=False)
    summary = {
        "run_dir": str(root),
        "methods": methods,
        "seeds": [int(seed) for seed in seeds],
        "scale_factor": scale,
        "synthetic_trips_per_run": n,
        "comparison_by_run": str(comparison_path),
        "comparison_compact": str(compact_path),
        "method_runs": method_summaries,
    }
    summary_path = root / "comparison_summary.json"
    write_json(summary_path, summary)
    summary["comparison_summary"] = str(summary_path)
    return MethodComparisonResult(run_dir=root, comparison=comparison, summary=summary)


def format_method_comparison_summary(result: MethodComparisonResult) -> str:
    lines = ["Synthesis method comparison complete", f"  run dir: {result.run_dir}"]
    if not result.comparison.empty:
        lines.append(f"  runs: {len(result.comparison)}")
        lines.append("  methods: " + ", ".join(result.comparison["synthesis_method"].astype(str).unique()))
        metric = "uncalibrated_absolute_proxy.spearman_correlation"
        if metric in result.comparison:
            best = result.comparison.sort_values(metric, ascending=False).iloc[0]
            lines.append(
                f"  best Spearman: {best['synthesis_method']} seed {best['seed']} = {best[metric]:.3f}"
            )
    lines.append(f"  comparison: {result.summary['comparison_by_run']}")
    lines.append(f"  compact: {result.summary['comparison_compact']}")
    return "\n".join(lines)
