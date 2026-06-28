import geopandas as gpd
from shapely.geometry import LineString

from tripsynth.data_sources.observed_counts.base import normalize_observed_counts


def test_normalize_observed_counts_case_insensitive_fields():
    raw = gpd.GeoDataFrame(
        {
            "OBJECTID": [1],
            "AADT": [1234],
            "COUNTY_CODE": [31],
            "F_SYSTEM": [2],
        },
        geometry=[LineString([(-77, 39), (-77.1, 39.1)])],
        crs="EPSG:4326",
    )
    normalized = normalize_observed_counts(
        raw,
        source="fhwa_hpms_2018",
        state="md",
        temporal_type="annual_average",
        default_year=2018,
        field_map={
            "segment_id": "objectid",
            "observed_aadt": "aadt",
            "county_fips_partial": "county_code",
            "functional_class": "f_system",
        },
    )
    row = normalized.gdf.iloc[0]
    assert row["observed_aadt"] == 1234
    assert row["county_fips"] == "24031"
    assert normalized.fields_mapped["observed_aadt"] == "AADT"
