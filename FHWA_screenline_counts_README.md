# FHWA TMAS screenline counts for June 10, 2018

This package is designed to produce `fhwa_2018_06_10_screenline_hourly_counts.csv` from FHWA TMAS data.

## Why the script needs a station-to-screenline map

FHWA TMAS provides continuous count **stations**, not TPB screenline definitions. To aggregate stations to screenlines, you need to assign the relevant count stations to your screenlines.

## Run

```bash
python build_fhwa_screenline_counts_2018_06_10.py
```

The first run downloads:

- `https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/2018/2018_station_data.zip`
- `https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/2018/jun_2018_ccs_data.zip`

It then writes:

- `outputs/fhwa_2018_06_10_all_station_hourly.csv`
- `outputs/fhwa_2018_06_10_station_metadata_parsed.csv`
- `outputs/fhwa_2018_06_10_station_daily_audit.csv`

Then fill `screenline_station_map.csv` with `include=1` for stations you want in each screenline and rerun.

## Final output

After station mapping, the main output is:

```text
outputs/fhwa_2018_06_10_screenline_hourly_counts.csv
```

Columns:

- `screenline`
- `date`
- `hour` (0-23)
- `observed_count`
- `n_station_lane_records`

## Recommended initial screenlines

- `dc_cordon`
- `dc_md`
- `dc_va_potomac`
- `montgomery_dc`
- `pg_dc`
- `nova_dc`

