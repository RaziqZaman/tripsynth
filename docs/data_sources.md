# Data Sources

## Census TIGER Tracts

The downloader reads `configs/data_sources.yaml`, downloads the configured 2020 TIGER tract ZIP files for Maryland and the District of Columbia, extracts them under `data/external/tiger_tracts/`, and validates that the shapefiles can be opened when GeoPandas is installed.

## MDOT SHA AADT/AAWDT

The MDOT SHA traffic count source is configured as an ArcGIS FeatureServer in `configs/data_sources.yaml`. The downloader queries service metadata, downloads the point layer, and attempts to discover a line or segment layer by layer ID and by names or descriptions containing `Line`, `Lines`, `Segment`, or `Segments`.

Expected outputs are:

```text
data/external/mdot_aadt/aadt_points.geojson
data/external/mdot_aadt/aadt_segments.geojson
data/external/mdot_aadt/aadt_service_metadata.json
data/external/DATA_MANIFEST.json
```

## Rerunning

```bash
python scripts/download_external_data.py --config configs/data_sources.yaml
```

If MDOT changes ArcGIS layer IDs, update `candidate_lines_layer_ids` or inspect `aadt_service_metadata.json` and choose the best matching layer. AADT validation will skip with a clear message if required files are missing or geospatial dependencies are unavailable.
