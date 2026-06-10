# MDOT I-TMS / AADT bulk run report
Run timestamp UTC: `2026-06-10T22:18:09Z`
Target date: `2018-06-10`
## Outputs
- `arcgis_aadt_locator/aadt_locations_export_bbox.csv` — candidate AADT/count-location inventory exported from the public AADT Locator web map, if ArcGIS access succeeded.
- `arcgis_aadt_locator/aadt_locations_export_bbox_map.html` — map of exported candidate locations.
- `itms_network_capture.jsonl` — captured I-TMS network calls from your browser session.
- `itms_endpoint_candidates.csv` — ranked captured endpoints to inspect/replay.
- `itms_hourly_availability_replay_results.csv` — replay results over input candidate locations, if a replayable GET endpoint was found.

## Summary
- AADT/locator rows exported: `6005`
- Replay rows checked: `0`

## Interpretation
This tool separates count-location inventory from verified hourly data. AADT/GIS points alone do not prove hourly availability. A location should be treated as hourly-usable only if I-TMS returns a valid hourly/volume report for the target date or MDOT provides the raw hourly records.

For county screenlines, keep only counters physically on or near the boundary/corridor being claimed and document missing crossings. Do not substitute I-270/I-495 approach stations for DC-Montgomery boundary crossings unless the screenline is explicitly described as a corridor proxy.
