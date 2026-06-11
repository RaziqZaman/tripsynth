from __future__ import annotations

from trip_synth.data.arcgis import discover_line_layer, esri_json_to_geojson, query_feature_layer


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

    def get(self, url, params=None, timeout=60):
        self.calls.append((url, params))
        if url.endswith("/0") and params == {"f": "json"}:
            return FakeResponse({"maxRecordCount": 1, "fields": [{"name": "OBJECTID"}]})
        if url.endswith("/0/query"):
            offset = params.get("resultOffset", 0)
            if offset == 0:
                return FakeResponse(
                    {
                        "features": [
                            {
                                "attributes": {"OBJECTID": 1},
                                "geometry": {"x": -76.0, "y": 39.0},
                            }
                        ],
                        "exceededTransferLimit": True,
                    }
                )
            return FakeResponse({"features": [], "exceededTransferLimit": False})
        return FakeResponse({"layers": []})


def test_discover_line_layer_by_name() -> None:
    metadata = {"layers": [{"id": 0, "name": "AADT Points"}, {"id": 2, "name": "AADT Segments"}]}
    assert discover_line_layer(metadata, []) == 2


def test_esri_json_to_geojson_point() -> None:
    geojson = esri_json_to_geojson(
        {"features": [{"attributes": {"id": 1}, "geometry": {"x": -76.0, "y": 39.0}}]}
    )
    assert geojson["type"] == "FeatureCollection"
    assert geojson["features"][0]["geometry"]["type"] == "Point"


def test_query_feature_layer_paginates_and_converts_json() -> None:
    session = FakeSession()
    geojson, metadata = query_feature_layer(
        "https://example.test/FeatureServer",
        0,
        page_size=1,
        prefer_geojson=False,
        session=session,
    )
    assert metadata["maxRecordCount"] == 1
    assert len(geojson["features"]) == 1
    assert len([call for call in session.calls if call[0].endswith("/query")]) == 2
