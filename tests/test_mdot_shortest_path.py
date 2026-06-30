import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from tripsynth.routing.mdot_shortest_path import route_mdot_shortest_paths
from tripsynth.validation.spatial_proxy import validate_spatial_aadt_proxy


def test_mdot_shortest_path_routes_connected_county_od():
    observed = gpd.GeoDataFrame(
        {
            "segment_id": ["a", "b"],
            "county_fips": ["24031", "24033"],
            "observed_aadt": [1000, 2000],
        },
        geometry=[
            LineString([(300000, 4300000), (310000, 4300000)]),
            LineString([(310000, 4300000), (320000, 4300000)]),
        ],
        crs="EPSG:26918",
    )
    trips = pd.DataFrame(
        {
            "origin_tract": ["24031000100", "24031000100"],
            "destination_tract": ["24033000100", "24033000100"],
        }
    )
    config = {
        "project": {"crs_projected": "EPSG:26918"},
        "validation": {"routing": {"mdot_shortest_path": {"snap_tolerance_m": 1}}},
    }
    routed = route_mdot_shortest_paths(trips, observed, config)
    assert routed.metadata["routes"] == 1
    assert routed.metadata["failed_routes"] == 0
    assert routed.routed_volumes["predicted_volume"].sum() > 0

    by_segment, metrics = validate_spatial_aadt_proxy(routed.routed_volumes, observed, config)
    assert by_segment["predicted_volume"].sum() > 0
    assert "direct_network_segment" in set(by_segment["match_method"])
    assert not metrics.empty


def test_mdot_shortest_path_routes_connected_tract_od(tmp_path):
    observed = gpd.GeoDataFrame(
        {
            "segment_id": ["a", "b"],
            "county_fips": ["24031", "24033"],
            "observed_aadt": [1000, 2000],
        },
        geometry=[
            LineString([(300000, 4300000), (310000, 4300000)]),
            LineString([(310000, 4300000), (320000, 4300000)]),
        ],
        crs="EPSG:26918",
    )
    tracts = gpd.GeoDataFrame(
        {
            "tract_fips": ["24031000100", "24033000100"],
            "county_fips": ["24031", "24033"],
        },
        geometry=[Point(300050, 4300000), Point(319950, 4300000)],
        crs="EPSG:26918",
    )
    tract_path = tmp_path / "tracts.geoparquet"
    tracts.to_parquet(tract_path, index=False)
    trips = pd.DataFrame(
        {
            "origin_tract": ["24031000100", "24031000100"],
            "destination_tract": ["24033000100", "24033000100"],
        }
    )
    config = {
        "project": {"crs_projected": "EPSG:26918"},
        "census": {"tracts": {"processed_path": str(tract_path)}},
        "validation": {
            "routing": {
                "mdot_shortest_path": {
                    "od_geography": "tract",
                    "auto_fetch_tracts": False,
                    "fallback_to_county_centroids": False,
                    "snap_tolerance_m": 1,
                }
            }
        },
    }

    routed = route_mdot_shortest_paths(trips, observed, config)

    assert routed.metadata["od_geography"] == "tract"
    assert routed.metadata["vehicle_trips_with_supported_tracts"] == 2
    assert routed.metadata["routes"] == 1
    assert routed.metadata["failed_routes"] == 0
    assert routed.route_table is not None
    assert len(routed.route_table) == 1
    assert routed.failed_routes is not None
    assert routed.failed_routes.empty
    assert routed.routed_volumes["predicted_volume"].sum() > 0


def test_mdot_shortest_path_can_prefer_high_aadt_corridor(tmp_path):
    observed = gpd.GeoDataFrame(
        {
            "segment_id": ["low_direct", "high_leg_1", "high_leg_2"],
            "county_fips": ["24031", "24031", "24031"],
            "observed_aadt": [10, 1000, 1000],
        },
        geometry=[
            LineString([(0, 0), (100, 0)]),
            LineString([(0, 0), (0, 10)]),
            LineString([(0, 10), (100, 0)]),
        ],
        crs="EPSG:26918",
    )
    tracts = gpd.GeoDataFrame(
        {
            "tract_fips": ["24031000100", "24031000200"],
            "county_fips": ["24031", "24031"],
        },
        geometry=[Point(0, 0), Point(100, 0)],
        crs="EPSG:26918",
    )
    tract_path = tmp_path / "tracts.geoparquet"
    tracts.to_parquet(tract_path, index=False)
    trips = pd.DataFrame(
        {
            "origin_tract": ["24031000100"],
            "destination_tract": ["24031000200"],
        }
    )
    config = {
        "project": {"crs_projected": "EPSG:26918"},
        "census": {"tracts": {"processed_path": str(tract_path)}},
        "validation": {
            "routing": {
                "mdot_shortest_path": {
                    "od_geography": "tract",
                    "auto_fetch_tracts": False,
                    "fallback_to_county_centroids": False,
                    "snap_tolerance_m": 1,
                    "edge_weight_strategy": "aadt_preferred",
                    "aadt_preference_alpha": 0.5,
                    "aadt_cost_min_multiplier": 0.25,
                    "aadt_cost_max_multiplier": 4.0,
                }
            }
        },
    }

    routed = route_mdot_shortest_paths(trips, observed, config)

    assert routed.metadata["edge_weight_strategy"] == "aadt_preferred"
    assert set(routed.routed_volumes["segment_id"]) == {"high_leg_1", "high_leg_2"}
