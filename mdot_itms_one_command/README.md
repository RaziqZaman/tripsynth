# MDOT SHA I-TMS / AADT one-command bulk audit

This project helps you decide whether MDOT SHA count locations are usable for county-traffic screenline validation on a target date, default **2018-06-10**.

It does three things:

1. Exports candidate AADT/count-location features from the official MDOT SHA AADT Locator ArcGIS web map when the ArcGIS layers are publicly queryable.
2. Opens the official MDOT SHA I-TMS public app in Chromium and captures network requests while you query hourly/volume reports.
3. Replays a discovered GET endpoint over `inputs/candidate_locations.csv`, if the captured endpoint is replayable, and writes CSV/map outputs.

## One command

### macOS / Linux

```bash
bash run.sh
```

### Windows PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File .\run_windows.ps1
```

The first run creates a virtual environment, installs Python packages, installs Chromium for Playwright, and starts the audit.

## What you do when the browser opens

The command opens the official I-TMS app. In that browser:

1. Search for a known count location, road, route, or county.
2. Enter the target date: `06/10/2018`.
3. Open one or more Volume/Hourly reports.
4. Close the browser window when finished.

The script saves the captured requests and tries to summarize/replay them automatically.

## Important: why one click is still needed

MDOT documents I-TMS, AADT Locator, Data Extractor, and Email Me, but does not publish a simple official bulk endpoint for all hourly reports. The safest reproducible approach is to capture the endpoint your browser actually uses, then replay it over the candidate locations.

This tool does not bypass access controls. It only records requests made by your browser while using the public app.

## Inputs

- `inputs/candidate_locations.csv` — default candidate list from the previously verified June 10, 2018 Maryland hourly locations. Replace or expand this with AADT Locator locations you want to test.
- `inputs/screenline_corridors.csv` — notes for PG-DC, Montgomery-DC, PG-Montgomery, and Maryland-DC screenline logic.
- `config.json` — date, ArcGIS item ID, bbox, and app URLs.

To change the target date:

```bash
bash run.sh --date 2018-06-10
```

To skip the browser and only export public AADT Locator GIS layers:

```bash
bash run.sh --skip-browser
```

## Outputs

The command writes to `outputs/`:

- `RUN_REPORT.md` — read this first.
- `arcgis_aadt_locator/aadt_layers_summary.csv` — discovered AADT Locator layers.
- `arcgis_aadt_locator/aadt_locations_export_bbox.csv` — exported AADT/count-location candidate features within the configured bbox.
- `arcgis_aadt_locator/aadt_locations_export_bbox_map.html` — interactive map of exported candidates.
- `itms_network_capture.jsonl` — captured I-TMS request/response metadata.
- `itms_endpoint_candidates.csv` — likely report endpoints ranked by relevance.
- `itms_hourly_availability_replay_results.csv` — bulk replay results if a GET endpoint was found.
- `itms_hourly_availability_replay_map.html` — replay result map if locations include coordinates.

## Screenline interpretation rules

For hourly validation, count locations should be treated as usable only when there is a valid hourly/volume report for the target date.

Use this hierarchy:

- Best: permanent/ATR continuous count locations with hourly volume on the target date.
- Conditional: portable/short-duration count locations only if actually counted on the target date.
- Not hourly-usable: AADT-only/factored segment locations without raw hourly reports.
- Still useful: AADT/AAWDT for daily or annual plausibility checks.

For county-traffic screenlines, the counter must also be physically meaningful for the claimed screenline. Do not count I-270 or I-495 stations as DC-Montgomery boundary crossings unless you explicitly label them as corridor proxies.

## Troubleshooting

If Playwright cannot install Chromium, run:

```bash
python -m playwright install chromium
```

If the AADT Locator export returns zero rows, edit `config.json` and widen `default_bbox`, or run the browser capture path.

If no replayable GET endpoint is found, inspect `outputs/itms_endpoint_candidates.csv`. The relevant report may use POST parameters; in that case, use the captured endpoint information or MDOT's Email Me service for an official bulk export.
