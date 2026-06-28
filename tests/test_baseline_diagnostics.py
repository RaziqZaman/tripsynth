from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from tripsynth.config import write_json
from tripsynth.reporting.baseline_diagnostics import run_baseline_diagnostics


def test_baseline_diagnostics_writes_summary_tables_and_figures(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "metrics").mkdir(parents=True)
    (run_dir / "tables").mkdir(parents=True)

    segments = gpd.GeoDataFrame(
        {
            "segment_id": ["a", "b", "c"],
            "county_fips": ["24031", "24031", "24033"],
            "route_name": ["A", "B", "C"],
            "observed_aadt": [1000.0, 500.0, 250.0],
            "predicted_volume": [800.0, 100.0, 300.0],
            "match_method": ["direct_network_segment", "unmatched", "direct_network_segment"],
        },
        geometry=[
            LineString([(-77.0, 39.0), (-77.01, 39.0)]),
            LineString([(-77.1, 39.0), (-77.11, 39.0)]),
            LineString([(-76.9, 38.9), (-76.91, 38.9)]),
        ],
        crs="EPSG:4326",
    )
    segment_path = run_dir / "metrics" / "metrics_by_segment.parquet"
    segments.to_parquet(segment_path, index=False)

    metrics = pd.DataFrame(
        [
            {
                "comparison_type": "uncalibrated_absolute_proxy",
                "pearson_correlation": 0.5,
                "spearman_correlation": 0.4,
                "jensen_shannon_divergence": 0.2,
                "top_10pct_overlap": 0.25,
            },
            {
                "comparison_type": "relative_share_validation",
                "share_pearson_correlation": 0.6,
                "share_spearman_correlation": 0.7,
                "share_jensen_shannon_divergence": 0.1,
                "share_top_10pct_overlap": 0.5,
            },
        ]
    )
    metrics_path = run_dir / "metrics" / "metrics_by_run.csv"
    metrics.to_csv(metrics_path, index=False)

    route_path = run_dir / "tables" / "route_table.csv"
    pd.DataFrame(
        [{"route_id": "24031000100_24033000100", "predicted_volume": 2.0, "synthetic_trips": 2}]
    ).to_csv(route_path, index=False)
    failed_path = run_dir / "tables" / "failed_routes.csv"
    pd.DataFrame(
        [{"route_id": "x_y", "failure_reason": "no_path", "predicted_volume": 1.0, "synthetic_trips": 1}]
    ).to_csv(failed_path, index=False)

    summary = {
        "run_dir": str(run_dir),
        "validation_mode": "spatial_aadt_proxy",
        "observed_counts": {"source": "mdot_sha_aadt", "records": 3},
        "vehicle_filter": {"input_trip_count": 10, "vehicle_trip_count": 8},
        "routing": {
            "od_geography": "tract",
            "vehicle_trips_with_supported_tracts": 5,
            "od_pairs": 3,
            "routes": 2,
            "failed_routes": 1,
            "routed_observed_segments": 2,
            "graph_nodes": 5,
            "graph_edges": 4,
        },
        "survey_coverage": {"first_survey_date": "2017-01-02", "last_survey_date": "2017-01-03"},
        "outputs": {
            "metrics_by_segment": str(segment_path),
            "metrics_by_run": str(metrics_path),
            "route_table": str(route_path),
            "failed_routes": str(failed_path),
        },
    }
    write_json(run_dir / "tables" / "baseline_summary.json", summary)

    result = run_baseline_diagnostics(run_dir, top_n=2)

    assert Path(result.output_paths["coverage_summary"]).exists()
    assert Path(result.output_paths["headline_metrics"]).exists()
    assert Path(result.output_paths["failed_routes"]).exists()
    assert Path(result.output_paths["observed_vs_predicted_png"]).exists()
    assert Path(result.output_paths["segment_residual_map_html"]).exists()
