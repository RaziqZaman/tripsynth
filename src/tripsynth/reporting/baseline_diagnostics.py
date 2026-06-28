"""Diagnostics and paper-ready artifacts for baseline runs."""

from __future__ import annotations

from dataclasses import dataclass
import html
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BaselineDiagnosticsResult:
    run_dir: Path
    output_paths: dict[str, str | None]
    summary: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator in (0, None) or not np.isfinite(denominator):
        return None
    return float(numerator / denominator)


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _load_metrics_by_segment(summary: dict[str, Any], run_dir: Path) -> gpd.GeoDataFrame:
    value = summary.get("outputs", {}).get("metrics_by_segment")
    path = Path(value) if value else run_dir / "metrics" / "metrics_by_segment.parquet"
    if not path.exists():
        raise FileNotFoundError(f"metrics_by_segment not found: {path}")
    return gpd.read_parquet(path)


def _load_metrics_by_run(summary: dict[str, Any], run_dir: Path) -> pd.DataFrame:
    value = summary.get("outputs", {}).get("metrics_by_run")
    path = Path(value) if value else run_dir / "metrics" / "metrics_by_run.csv"
    if not path.exists():
        raise FileNotFoundError(f"metrics_by_run not found: {path}")
    return pd.read_csv(path)


def _optional_table(summary: dict[str, Any], run_dir: Path, key: str, fallback_name: str) -> pd.DataFrame:
    value = summary.get("outputs", {}).get(key)
    path = Path(value) if value else run_dir / "tables" / fallback_name
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _coverage_summary(summary: dict[str, Any]) -> pd.DataFrame:
    routing = summary.get("routing", {})
    vehicle = summary.get("vehicle_filter", {})
    observed = summary.get("observed_counts", {})
    survey = summary.get("survey_coverage", {})
    vehicle_trips = int(vehicle.get("vehicle_trip_count", 0) or 0)
    supported = int(
        routing.get("vehicle_trips_with_supported_tracts")
        or routing.get("vehicle_trips_with_supported_counties")
        or 0
    )
    od_pairs = int(routing.get("od_pairs", 0) or 0)
    routes = int(routing.get("routes", 0) or 0)
    failed = int(routing.get("failed_routes", 0) or 0)
    row = {
        "run_dir": summary.get("run_dir"),
        "validation_mode": summary.get("validation_mode"),
        "observed_source": observed.get("source"),
        "observed_records": observed.get("records"),
        "od_geography": routing.get("od_geography"),
        "input_trip_count": vehicle.get("input_trip_count"),
        "vehicle_trip_count": vehicle_trips,
        "supported_vehicle_trip_count": supported,
        "supported_vehicle_trip_share": _safe_divide(supported, vehicle_trips),
        "od_pairs": od_pairs,
        "routes": routes,
        "failed_routes": failed,
        "route_success_rate": _safe_divide(routes, od_pairs),
        "routed_observed_segments": routing.get("routed_observed_segments"),
        "graph_nodes": routing.get("graph_nodes"),
        "graph_edges": routing.get("graph_edges"),
        "first_survey_date": survey.get("first_survey_date"),
        "last_survey_date": survey.get("last_survey_date"),
    }
    return pd.DataFrame([row])


def _headline_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: dict[str, Any] = {}
    for record in metrics.to_dict("records"):
        label = record.get("comparison_type")
        if not label:
            continue
        prefix = str(label)
        for key, value in record.items():
            if key == "comparison_type" or pd.isna(value):
                continue
            rows[f"{prefix}.{key}"] = value
    keep = [
        "uncalibrated_absolute_proxy.pearson_correlation",
        "uncalibrated_absolute_proxy.spearman_correlation",
        "uncalibrated_absolute_proxy.jensen_shannon_divergence",
        "uncalibrated_absolute_proxy.top_10pct_overlap",
        "sum_calibrated_absolute_proxy.bias_ratio",
        "relative_share_validation.share_pearson_correlation",
        "relative_share_validation.share_spearman_correlation",
        "relative_share_validation.share_jensen_shannon_divergence",
        "relative_share_validation.share_top_10pct_overlap",
    ]
    ordered = {key: rows.get(key) for key in keep if key in rows}
    for key, value in rows.items():
        ordered.setdefault(key, value)
    return pd.DataFrame([ordered])


def _segment_working_frame(segments: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    frame = segments.copy()
    frame["observed_aadt"] = _clean_numeric(frame.get("observed_aadt", pd.Series(index=frame.index)))
    frame["predicted_volume"] = _clean_numeric(frame.get("predicted_volume", pd.Series(index=frame.index)))
    frame["absolute_error"] = (frame["predicted_volume"] - frame["observed_aadt"]).abs()
    frame["signed_error"] = frame["predicted_volume"] - frame["observed_aadt"]
    frame["prediction_to_observed_ratio"] = np.where(
        frame["observed_aadt"] > 0,
        frame["predicted_volume"] / frame["observed_aadt"],
        np.nan,
    )
    pred_sum = float(frame["predicted_volume"].sum())
    obs_sum = float(frame["observed_aadt"].sum())
    frame["predicted_share"] = frame["predicted_volume"] / pred_sum if pred_sum > 0 else 0.0
    frame["observed_share"] = frame["observed_aadt"] / obs_sum if obs_sum > 0 else 0.0
    frame["share_residual"] = frame["predicted_share"] - frame["observed_share"]
    return frame


def _table_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "segment_id",
        "route_id",
        "route_name",
        "county_fips",
        "functional_class",
        "through_lanes",
        "observed_aadt",
        "predicted_volume",
        "absolute_error",
        "signed_error",
        "prediction_to_observed_ratio",
        "predicted_share",
        "observed_share",
        "share_residual",
        "match_method",
    ]
    return [column for column in preferred if column in frame.columns]


def _county_summary(frame: gpd.GeoDataFrame) -> pd.DataFrame:
    if "county_fips" not in frame:
        return pd.DataFrame()
    grouped = (
        frame.groupby("county_fips", dropna=False)
        .agg(
            segments=("segment_id", "size") if "segment_id" in frame else ("predicted_volume", "size"),
            routed_segments=("predicted_volume", lambda values: int((values > 0).sum())),
            observed_aadt_sum=("observed_aadt", "sum"),
            predicted_volume_sum=("predicted_volume", "sum"),
            absolute_error_sum=("absolute_error", "sum"),
        )
        .reset_index()
    )
    grouped["bias_ratio"] = np.where(
        grouped["observed_aadt_sum"] > 0,
        grouped["predicted_volume_sum"] / grouped["observed_aadt_sum"],
        np.nan,
    )
    total_observed = float(grouped["observed_aadt_sum"].sum())
    total_predicted = float(grouped["predicted_volume_sum"].sum())
    grouped["observed_share"] = grouped["observed_aadt_sum"] / total_observed if total_observed > 0 else 0.0
    grouped["predicted_share"] = grouped["predicted_volume_sum"] / total_predicted if total_predicted > 0 else 0.0
    grouped["share_residual"] = grouped["predicted_share"] - grouped["observed_share"]
    return grouped.sort_values("absolute_error_sum", ascending=False)


def _failure_summary(failed_routes: pd.DataFrame) -> pd.DataFrame:
    if failed_routes.empty:
        return pd.DataFrame([{"failure_reason": "none_or_not_available", "routes": 0, "synthetic_trips": 0}])
    synthetic = "synthetic_trips" if "synthetic_trips" in failed_routes else "predicted_volume"
    return (
        failed_routes.groupby("failure_reason", dropna=False)
        .agg(routes=("failure_reason", "size"), synthetic_trips=(synthetic, "sum"))
        .reset_index()
        .sort_values("routes", ascending=False)
    )


def _write_scatter(frame: gpd.GeoDataFrame, path: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot = frame.loc[(frame["observed_aadt"] > 0) | (frame["predicted_volume"] > 0)].copy()
        if plot.empty:
            return None
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(
            plot["observed_aadt"],
            plot["predicted_volume"],
            s=10,
            alpha=0.45,
            edgecolors="none",
        )
        max_value = float(max(plot["observed_aadt"].max(), plot["predicted_volume"].max()))
        if max_value > 0:
            ax.plot([1, max_value], [1, max_value], color="#444444", linewidth=1, linestyle="--")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel("Observed AADT")
        ax.set_ylabel("Predicted routed volume")
        ax.set_title("Observed vs predicted segment volume")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return str(path)
    except Exception:
        return None


def _residual_color(value: float) -> str:
    if not np.isfinite(value):
        return "#7f8c8d"
    if value <= -0.001:
        return "#2166ac"
    if value < -0.0002:
        return "#67a9cf"
    if value <= 0.0002:
        return "#bdbdbd"
    if value < 0.001:
        return "#ef8a62"
    return "#b2182b"


def _write_residual_map(frame: gpd.GeoDataFrame, path: Path) -> str | None:
    if frame.empty:
        return None
    map_gdf = frame.to_crs("EPSG:4326").copy()
    map_gdf["residual_color"] = map_gdf["share_residual"].map(_residual_color)
    keep = [
        column
        for column in [
            "segment_id",
            "route_name",
            "county_fips",
            "observed_aadt",
            "predicted_volume",
            "share_residual",
            "residual_color",
            "geometry",
        ]
        if column in map_gdf.columns
    ]
    geojson = json.loads(map_gdf[keep].to_json())
    bounds = map_gdf.total_bounds
    center_lat = float((bounds[1] + bounds[3]) / 2)
    center_lon = float((bounds[0] + bounds[2]) / 2)
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Baseline Segment Diagnostics</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; }}
    #map {{ width: 100vw; height: 100vh; }}
    .panel {{ position: absolute; top: 12px; left: 56px; z-index: 500; background: white; padding: 8px 10px; border: 1px solid #999; max-width: 420px; font-size: 13px; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="panel">
    <strong>Baseline Segment Diagnostics</strong><br>
    Blue: under-represented share; red: over-represented share.<br>
    Features: {len(map_gdf):,}
  </div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const data = {json.dumps(geojson)};
    const map = L.map("map").setView([{center_lat:.6f}, {center_lon:.6f}], 9);
    L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{ maxZoom: 18, attribution: '&copy; OpenStreetMap contributors' }}).addTo(map);
    const layer = L.geoJSON(data, {{
      style: feature => {{
        const color = feature.properties.residual_color || "#7f8c8d";
        return {{ color, weight: 2, opacity: 0.8 }};
      }},
      onEachFeature: (feature, layer) => {{
        const p = feature.properties;
        layer.bindPopup(`<b>${{p.route_name || p.segment_id || "segment"}}</b><br>County: ${{p.county_fips || "missing"}}<br>Observed AADT: ${{Number(p.observed_aadt || 0).toFixed(1)}}<br>Predicted: ${{Number(p.predicted_volume || 0).toFixed(1)}}<br>Share residual: ${{Number(p.share_residual || 0).toExponential(3)}}`);
      }}
    }}).addTo(map);
    map.fitBounds(layer.getBounds());
  </script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")
    return str(path)


def run_baseline_diagnostics(run_dir: str | Path, *, top_n: int = 25) -> BaselineDiagnosticsResult:
    run_path = Path(run_dir)
    summary_path = run_path / "tables" / "baseline_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Baseline summary not found: {summary_path}")
    summary = _read_json(summary_path)
    segments = _segment_working_frame(_load_metrics_by_segment(summary, run_path))
    metrics = _load_metrics_by_run(summary, run_path)
    route_table = _optional_table(summary, run_path, "route_table", "route_table.csv")
    failed_routes = _optional_table(summary, run_path, "failed_routes", "failed_routes.csv")

    tables_dir = run_path / "diagnostics" / "tables"
    figures_dir = run_path / "diagnostics" / "figures"
    maps_dir = run_path / "diagnostics" / "maps"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    coverage = _coverage_summary(summary)
    headline = _headline_metrics(metrics)
    county = _county_summary(segments)
    cols = _table_columns(segments)
    top_predicted = segments.sort_values("predicted_volume", ascending=False).head(top_n)[cols]
    largest_errors = segments.sort_values("absolute_error", ascending=False).head(top_n)[cols]
    top_observed = segments.sort_values("observed_aadt", ascending=False).head(top_n)[cols]
    failure_summary = _failure_summary(failed_routes)

    output_paths: dict[str, str | None] = {}
    artifacts = {
        "coverage_summary": (coverage, tables_dir / "coverage_summary.csv"),
        "headline_metrics": (headline, tables_dir / "headline_metrics.csv"),
        "metrics_by_run_copy": (metrics, tables_dir / "metrics_by_run.csv"),
        "county_segment_summary": (county, tables_dir / "county_segment_summary.csv"),
        "top_predicted_segments": (top_predicted, tables_dir / "top_predicted_segments.csv"),
        "top_observed_segments": (top_observed, tables_dir / "top_observed_segments.csv"),
        "largest_absolute_errors": (largest_errors, tables_dir / "largest_absolute_errors.csv"),
        "failed_route_summary": (failure_summary, tables_dir / "failed_route_summary.csv"),
    }
    for key, (frame, path) in artifacts.items():
        frame.to_csv(path, index=False)
        output_paths[key] = str(path)

    if not route_table.empty:
        route_path = tables_dir / "top_route_paths.csv"
        sort_col = "predicted_volume" if "predicted_volume" in route_table else route_table.columns[0]
        route_table.sort_values(sort_col, ascending=False).head(top_n).to_csv(route_path, index=False)
        output_paths["top_route_paths"] = str(route_path)
    else:
        output_paths["top_route_paths"] = None

    if not failed_routes.empty:
        failed_path = tables_dir / "failed_routes.csv"
        failed_routes.sort_values("predicted_volume", ascending=False).to_csv(failed_path, index=False)
        output_paths["failed_routes"] = str(failed_path)
    else:
        output_paths["failed_routes"] = None

    output_paths["observed_vs_predicted_png"] = _write_scatter(
        segments, figures_dir / "observed_vs_predicted.png"
    )
    output_paths["segment_residual_map_html"] = _write_residual_map(
        segments, maps_dir / "segment_residual_map.html"
    )

    diagnostics_summary = {
        "run_dir": str(run_path),
        "top_n": int(top_n),
        "coverage": coverage.to_dict("records")[0] if not coverage.empty else {},
        "outputs": output_paths,
    }
    summary_out = run_path / "diagnostics" / "diagnostics_summary.json"
    _write_json(summary_out, diagnostics_summary)
    output_paths["diagnostics_summary"] = str(summary_out)
    return BaselineDiagnosticsResult(run_dir=run_path, output_paths=output_paths, summary=diagnostics_summary)


def format_diagnostics_summary(result: BaselineDiagnosticsResult) -> str:
    coverage = result.summary.get("coverage", {})
    lines = ["Baseline diagnostics complete", f"  run dir: {result.run_dir}"]
    for key in [
        "vehicle_trip_count",
        "supported_vehicle_trip_count",
        "od_pairs",
        "routes",
        "failed_routes",
        "route_success_rate",
        "routed_observed_segments",
    ]:
        if key in coverage:
            lines.append(f"  {key}: {coverage[key]}")
    lines.append("  key outputs:")
    for key in [
        "coverage_summary",
        "headline_metrics",
        "county_segment_summary",
        "top_predicted_segments",
        "largest_absolute_errors",
        "failed_routes",
        "observed_vs_predicted_png",
        "segment_residual_map_html",
    ]:
        value = result.output_paths.get(key)
        if value:
            lines.append(f"    {key}: {value}")
    return "\n".join(lines)
