# Adjacent Census Tract AADT Station Screenlines

This project builds a station-level validation layer for **adjacent Census tract inflow/outflow daily count validation** using MDOT SHA AADT count locations.

The output is not a county cordon. It is a set of micro-screenlines: each accepted MDOT AADT count location is assigned to the Census tract containing the station and the nearest adjacent tract across the closest tract boundary. That tract pair becomes the candidate daily screenline.

## One command

```bash
bash run.sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_windows.ps1
```

The first run downloads Census TIGER/Line 2019 tract shapefiles for Maryland and DC from the official Census TIGER directory, then writes outputs into `outputs/`.

## Main outputs

After `bash run.sh` finishes, start with:

```text
outputs/RUN_REPORT.md
outputs/tract_pair_aadt_station_screenlines.csv
outputs/tract_pair_screenlines_map.html
outputs/tract_pair_screenline_summary_by_group.csv
outputs/tract_pair_screenline_validation_template.csv
outputs/rejected_or_ambiguous_station_candidates.csv
```

## What the screenline CSV means

Each accepted row is a candidate station-level tract-pair screenline:

```text
screenline_id
LOCATION_ID
screenline_family
tract_a_geoid
tract_b_geoid
station_tract_geoid
adjacent_tract_geoid
AADT_2017
AADT_2018
AAWDT_2017
AAWDT_2018
assignment_quality
station_to_shared_boundary_m
shared_boundary_length_m
```

The `assignment_quality` field is deliberately conservative:

- `high`: station is very close to a shared tract boundary and the adjacent tract assignment is not crowded/ambiguous.
- `medium`: usable but review visually.
- `low_review`: possible but should be manually inspected.
- `reject_*`: not used for the screenline output.

## Recommended validation use

Use `AADT_2017` and `AADT_2018` for daily observed traffic. Compare these to **daily modeled/survey vehicle flows** crossing the tract-pair screenline.

The included comparison script computes a strict direct-adjacent-OD comparison:

```bash
bash run_compare_survey_example.sh
```

That direct-OD comparison is intentionally conservative and will undercount through trips. A more rigorous version should count all modeled OD paths that cross the tract-pair boundary geometry, not just trips whose origin and destination are the two adjacent tracts.

## Data sources

- MDOT SHA AADT Locator export supplied by the user in this conversation; exported here as `inputs/mdot_aadt_locator_full_export.csv`.
- Census TIGER/Line 2019 tract shapefiles for Maryland and DC, downloaded by `src/build_adjacent_tract_screenlines.py`.

## Key methodological caveat

AADT count locations provide roadway segment traffic exposure. A tract-pair screenline test is most defensible when the station lies close to a tract boundary and the modeled flow is counted as boundary-crossing vehicle trips. Direct adjacent tract-to-tract OD is stricter and can undercount through traffic.
