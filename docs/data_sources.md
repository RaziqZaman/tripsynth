# Data Sources

## Census TIGER Tracts

The downloader reads `configs/data_sources.yaml`, downloads the configured 2019 TIGER tract ZIP files for Maryland and the District of Columbia, extracts them under `data/external/tiger_tracts/`, and validates that the shapefiles can be opened when GeoPandas is installed. The run configs point to the 2019 tract vintage for the 2017-2019 survey window.

## MDOT SHA AADT Station Locations

The MDOT SHA traffic count station source is configured as an ArcGIS FeatureServer in `configs/data_sources.yaml`. The downloader queries service metadata, downloads the point layer, and attempts to discover a line or segment layer by layer ID and by names or descriptions containing `Line`, `Lines`, `Segment`, or `Segments`.

Expected station-location outputs are:

```text
data/external/mdot_aadt/aadt_points.geojson
data/external/mdot_aadt/aadt_segments.geojson
data/external/mdot_aadt/aadt_service_metadata.json
data/external/DATA_MANIFEST.json
```

## Two-Prong Count Inputs

The default AADT validation mode is now `two_prong`. It uses annual-average station or line volume sources for dense spatial validation and FHWA TMAS station-hour counts for sparse temporal validation.

Configured count inputs:

```yaml
aadt_validation:
  validation_mode: two_prong
  annual_comparison_basis: average_day
  daily_counts:
    file: data/external/mdot_daily_counts/station_daily_counts.csv
    station_id_column: station_id
    date_column: date
    count_column: observed_count
  hourly_counts:
    file: data/external/mdot_daily_counts/station_hourly_counts.csv
    station_id_column: station_id
    date_column: date
    hour_column: hour
    count_column: observed_count
```

The dense annual tier uses MDOT AADT/AAWDT screenline station totals. The hourly tier uses FHWA TMAS `station_hourly_counts.csv` and scales sparse FHWA station counts by each matched station twin annual-average share of total observed screenline traffic.

Legacy `validation_mode: exact_day` still reads `station_daily_counts.csv` directly, but it is no longer the default external validation design because FHWA TMAS station coverage is too sparse for dense screenlines.

## VDOT And DDOT Traffic Volume Reconnaissance

VDOT has dense public bidirectional traffic-volume line layers for the relevant era. The 2018 and 2019 ArcGIS layers expose `ADT`, `AAWDT`, quality codes, route identifiers, and line geometry. The layer descriptions identify these as average daily traffic / annual average weekday traffic products, updated annually, not station-day or hourly count tables.

Useful VDOT links:

- Virginia Roads Traffic Volume hub: https://www.virginiaroads.org/datasets/VDOT::traffic-volume/about
- VDOT 2018 service layer: https://services.arcgis.com/p5v98VHDX9Atv3l7/arcgis/rest/services/Traffic_Volume_ADT_2018/FeatureServer/0
- VDOT 2019 service layer: https://services.arcgis.com/p5v98VHDX9Atv3l7/arcgis/rest/services/VDOT_Traffic_Volume_2019a/FeatureServer/0

DDOT has dense year-specific traffic-volume roadway-block layers for 2017, 2018, and 2019. Their service descriptions say counts are collected in both directions at preselected HPMS locations on a three-year cycle and converted to Annual Average Daily Traffic. DDOT also publishes 55 traffic monitoring station locations, but the public station layer checked here contains station metadata rather than daily/hourly count totals.

Useful DDOT links:

- DDOT 2017 traffic volume: https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Transportation_TrafficVolume_WebMercator/MapServer/134
- DDOT 2018 traffic volume: https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Transportation_TrafficVolume_WebMercator/MapServer/154
- DDOT 2019 traffic volume: https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Transportation_TrafficVolume_WebMercator/MapServer/167
- DDOT traffic monitoring stations: https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Transportation_Sensors_WebMercator/MapServer/92

Methodological implication: VDOT and DDOT can help with dense annual-average spatial validation around Virginia and DC, but the public sources found so far do not replace missing MDOT exact-day station totals. The temporal validation source remains FHWA TMAS continuous-count data, which is high-resolution in time but sparse in space.

## Rerunning

```bash
python scripts/download_external_data.py --config configs/data_sources.yaml
```

If MDOT changes ArcGIS layer IDs, update `candidate_lines_layer_ids` or inspect `aadt_service_metadata.json` and choose the best matching layer. AADT validation will skip with a clear message if required files are missing or geospatial dependencies are unavailable.
