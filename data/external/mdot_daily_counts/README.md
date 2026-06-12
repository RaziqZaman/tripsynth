# MDOT/FHWA Count Inputs

This directory contains the FHWA TMAS temporal count inputs used by the two-prong AADT screenline validator.

## Files

- `station_daily_counts.csv`: one station-date row per Maryland continuous/permanent counter, with 24-hour observed vehicle volume.
- `station_hourly_counts.csv`: one station-date-hour row per Maryland continuous/permanent counter, used by the hourly TMAS validation tier.
- `fhwa_tmas_direction_lane_daily_counts_2017-10-01_2019-07-31.csv`: direction/lane audit table after duplicate consolidation; includes `source_record_count`, `duplicate_policy`, and `hourly_counts`.
- `fhwa_tmas_station_metadata_2017_2019.csv`: FHWA station metadata rows for Maryland for 2017, 2018, and 2019.
- `fhwa_tmas_daily_counts_manifest.json`: source URLs, pull timestamp, row counts, and duplicate-handling metadata.
- `itms_station_report_index_2017-10-01_2019-07-31.csv`: official MDOT I-TMS report inventory rows for mapped station IDs. This is provenance for count reports, not observed station-day totals.
- `itms_station_report_index_manifest.json`: source and date-window metadata for the I-TMS index.

Build summary from `fhwa_tmas_daily_counts_manifest.json`:

- 37,800 station-day rows
- 907,200 station-hour rows
- 64 continuous-count stations
- 2017-10-01 to 2019-07-31
- 75,600 unique direction/lane-day rows
- 79,087 raw direction/lane-day rows before duplicate consolidation
- 3,487 duplicate station/date/direction/lane groups collapsed by taking the maximum submitted daily total within the duplicate group
- all retained direction/lane records have 24 observed hourly bins

## Station ID Convention

FHWA IDs such as `0P0009` are normalized to MDOT-style IDs such as `P0009`, matching the MDOT AADT Locator `LOCATION_ID` convention used by `geo/screenline_station_map.parquet`.

## Validation Use

The current default AADT validation mode is `two_prong`:

1. Dense annual-average screenline validation compares synthetic long-term average daily screenline crossings to observed annual-average screenline totals from the dense MDOT station set.
2. Sparse hourly validation compares synthetic hourly crossings to FHWA TMAS station-hour counts. Because FHWA TMAS stations are sparse, each matched station is scaled by its annual-average share of the total observed screenline volume.

## Source Links

FHWA TMAS data archive:

https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/

MDOT SHA public I-TMS:

https://maps.roads.maryland.gov/itms_public/

MDOT SHA AADT FeatureServer used for station point locations and annual-average IDs/count fields:

https://mdgeodata.md.gov/imap/rest/services/Transportation/MD_AnnualAverageDailyTraffic/FeatureServer/0?f=pjson

## Coverage Note

These FHWA files cover Maryland continuous/permanent count stations that appear in TMAS monthly continuous count submissions, not every short-duration or program-count report visible in I-TMS. The hourly tier is therefore intentionally sparse and scaled by the dense annual MDOT screenline station set.
