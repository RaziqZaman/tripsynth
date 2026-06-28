from pathlib import Path

from tripsynth.data_sources.observed_counts.arcgis import ArcGISFeatureServerClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append((url, params))
        if url.endswith("FeatureServer"):
            return FakeResponse({"layers": [{"id": 0}], "copyrightText": "public"})
        if url.endswith("FeatureServer/0"):
            return FakeResponse({"maxRecordCount": 1, "fields": [{"name": "AADT"}]})
        offset = params["resultOffset"]
        if offset >= 2:
            return FakeResponse({"type": "FeatureCollection", "features": []})
        return FakeResponse(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"OBJECTID": offset + 1, "AADT": 100 + offset},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[-77.0, 38.9], [-77.01, 38.91]],
                        },
                    }
                ],
            }
        )


def test_arcgis_paginates_and_caches(tmp_path):
    client = ArcGISFeatureServerClient(session=FakeSession())
    gdf, service_meta, layer_meta = client.download_layer(
        "https://example.test/FeatureServer",
        layer=0,
        cache_dir=tmp_path,
    )
    assert len(gdf) == 2
    assert service_meta["copyrightText"] == "public"
    assert layer_meta["maxRecordCount"] == 1
    assert (Path(tmp_path) / "page_00000.geojson").exists()
    assert (Path(tmp_path) / "page_00001.geojson").exists()
