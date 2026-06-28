import geopandas as gpd
from shapely.geometry import LineString

from tripsynth.data_sources.observed_counts.mdot_sha import fetch_mdot_sha_aadt


class FakeMDOTClient:
    def download_layer(self, url, *, layer, cache_dir, result_record_count=None):
        raw = gpd.GeoDataFrame(
            {
                "OBJECTID": [1, 2, 3],
                "COUNTY_DESC": ["Montgomery", "Prince George's", "Baltimore"],
                "ROADNAME": ["Road A", "Road B", "Road C"],
                "ROUTEID_RH": ["a", "b", "c"],
                "AADT_2018": [1000, 2000, 3000],
                "F_SYSTEM": [2, 3, 4],
                "NUM_LANES": [2, 4, 2],
            },
            geometry=[
                LineString([(-77.1, 39.0), (-77.0, 39.1)]),
                LineString([(-76.9, 38.9), (-76.8, 39.0)]),
                LineString([(-76.7, 39.2), (-76.6, 39.3)]),
            ],
            crs="EPSG:4326",
        )
        return raw, {"copyrightText": "MDOT public"}, {"maxRecordCount": 1000}


def test_fetch_mdot_sha_maps_county_names_to_fips_and_filters(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {
        "study_area": {"counties": {"md": ["24031", "24033"]}},
        "observed_counts": {
            "sources": {
                "mdot_sha_aadt": {
                    "enabled": True,
                    "url": "https://example.test/FeatureServer",
                    "layer": 1,
                    "temporal_type": "annual_average",
                    "field_map": {
                        "segment_id": "OBJECTID",
                        "route_id": "ROUTEID_RH",
                        "route_name": "ROADNAME",
                        "observed_aadt": "AADT_2018",
                        "functional_class": "F_SYSTEM",
                        "through_lanes": "NUM_LANES",
                    },
                }
            }
        },
    }
    result = fetch_mdot_sha_aadt(config, client=FakeMDOTClient())
    assert len(result.filtered) == 2
    assert set(result.filtered["county_fips"]) == {"24031", "24033"}
    assert result.filtered["observed_aadt"].sum() == 3000
    assert result.output_paths["filtered_geoparquet"]
