import geopandas as gpd
from shapely.geometry import LineString

from tripsynth.data_sources.observed_counts.fhwa_hpms import fetch_fhwa_hpms_2018


class FakeHPMSClient:
    def __init__(self):
        self.calls = 0

    def download_layer(self, url, *, layer, cache_dir):
        self.calls += 1
        state = "md" if "Maryland" in url else "dc"
        county = 31 if state == "md" else 1
        raw = gpd.GeoDataFrame(
            {
                "objectid": [self.calls],
                "aadt": [1000 * self.calls],
                "county_code": [county],
                "year_record": [2018],
                "route_id": [f"r{self.calls}"],
                "route_name": [f"Route {self.calls}"],
                "f_system": [2],
            },
            geometry=[LineString([(-77.0, 38.9), (-77.01, 38.91)])],
            crs="EPSG:4326",
        )
        return raw, {"copyrightText": "public"}, {"maxRecordCount": 1000}


def test_fetch_counts_writes_outputs_and_uses_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {
        "study_area": {"counties": {"md": ["24031"]}},
        "observed_counts": {
            "sources": {
                "fhwa_hpms_2018": {
                    "enabled": True,
                    "temporal_type": "annual_average",
                    "states": {
                        "md": {
                            "url": "https://example.test/Maryland_2018_PR/FeatureServer",
                            "layer": 0,
                        }
                    },
                    "field_map": {
                        "year": "year_record",
                        "segment_id": "objectid",
                        "route_id": "route_id",
                        "route_name": "route_name",
                        "observed_aadt": "aadt",
                        "county_fips_partial": "county_code",
                        "functional_class": "f_system",
                    },
                }
            }
        },
    }
    client = FakeHPMSClient()
    result = fetch_fhwa_hpms_2018(config, client=client)
    assert len(result.filtered) == 1
    assert result.filtered.iloc[0]["county_fips"] == "24031"
    assert client.calls == 1
    assert result.output_paths["filtered_geoparquet"]
    assert (tmp_path / "data/metadata/observed_counts_manifest.json").exists()
    assert (tmp_path / "reports/tables/observed_counts_sample.csv").exists()

    cached = fetch_fhwa_hpms_2018(config, client=client)
    assert cached.used_cache is True
    assert client.calls == 1
