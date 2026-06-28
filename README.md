# Trip Synthesis

Trip Synthesis is a reproducible research-code pipeline for augmenting limited household travel-behavior observations, routing the synthesized trips, and validating the routed roadway volumes against observed traffic-count proxies.

The repository is designed for a TRB-quality experimental workflow, but it starts from a careful premise: the local 2017-2019 household travel survey may not cover every month, day, or time period in those years. The first pipeline step audits the survey's actual temporal coverage before any synthesis, routing, or validation claim is made.

## Baseline Validation

The external validation mode is **spatial roadway-volume proxy validation against AADT**. The default source is MDOT SHA AADT for Maryland-only validation; FHWA HPMS 2018 remains available as an optional source. AADT is useful for asking whether a synthesis-routing pipeline reproduces spatial patterns of roadway demand; it is not same-day, same-month, or exact temporal ground truth for every survey trip.

Temporal validation is opt-in and guarded. It only runs when the survey audit and the observed-count dataset both expose overlapping dates or months.

## Required Local Input

Configure the household travel survey in `configs/default.yaml`. The current default points at the local transformed survey:

```yaml
survey:
  path: 00_survey-transformed.csv
```

Schema mapping is explicit. The default config derives `trip_date` from `tdate_week` and `tdate_dow`: `tdate_week` is weeks after January 1, 2017, and `tdate_dow` uses 1=Monday through 7=Sunday. The transformed `year` column appears to be vehicle model year, so it is intentionally not used as survey year.

## First Commands

Run the survey audit:

```bash
PYTHONPATH=src .venv/bin/python -m tripsynth.cli audit-survey --config configs/default.yaml
```

Fetch public observed counts from MDOT SHA AADT for Maryland-only external validation:

```bash
PYTHONPATH=src .venv/bin/python -m tripsynth.cli fetch-counts --config configs/default.yaml
```

The default fetch step downloads the official MDOT SHA Annual Average Daily Traffic ArcGIS FeatureServer, normalizes to the common observed-count schema, filters to the configured Maryland study-area counties, and writes manifests, sample tables, and coverage maps. FHWA HPMS can still be retried with `--source fhwa_hpms_2018`.

Official MDOT SHA AADT source: https://mdgeodata.md.gov/imap/rest/services/Transportation/MD_AnnualAverageDailyTraffic/FeatureServer

Prepare Census TIGER/Line tract geometries for tract-level OD routing:

```bash
PYTHONPATH=src .venv/bin/python -m tripsynth.cli prep-network --config configs/default.yaml
```

The tract prep step downloads 2018 TIGER/Line tracts for the configured DC, Maryland, and Virginia study-area states, filters them to the configured study counties, and caches `data/processed/census/tiger_tracts_study_area.geoparquet`. The MDOT validation route assignment then uses 11-digit origin and destination tract FIPS, restricted to tracts in the Maryland counties covered by MDOT SHA AADT.

Run Maryland-only external validation after fetching counts and preparing tracts:

```bash
PYTHONPATH=src .venv/bin/python -m tripsynth.cli run-baseline --config configs/default.yaml --validation-mode spatial_aadt_proxy
```

Run the currently implemented baseline without observed counts using survey-internal holdout validation:

```bash
PYTHONPATH=src .venv/bin/python -m tripsynth.cli run-baseline --config configs/default.yaml
```

The same baseline can also be launched through the experiment entry point:

```bash
PYTHONPATH=src .venv/bin/python -m tripsynth.cli run-experiment --config configs/default.yaml --baseline-only
```

By default this first baseline uses weighted resampling and validates synthetic trips against a held-out survey sample, so it does not require FHWA/HPMS counts. To run count-based validation, pass `--validation-mode spatial_aadt_proxy`; that mode now uses MDOT SHA AADT by default, filters to configured auto modes, routes 11-digit tract OD volumes over the MDOT AADT line graph with distance-weighted shortest paths, and validates Maryland-only spatial roadway-volume patterns against AADT.

If the FHWA FeatureServer is unavailable and you still want AADT validation, configure a local observed-count file under `observed_counts.sources.local_user_counts` with `enabled: true`, a `path`, and a `field_map` for `observed_aadt`, `segment_id`, `county_fips`, and `geometry` or `lat`/`lon`. CSV files with WKT geometry are supported.

## Smoke Test

```bash
make smoke
```

The tests use small synthetic data and mocked ArcGIS responses; they do not require large downloads.

## Outputs

Survey audit outputs:

- `data/metadata/survey_audit.json`
- `reports/tables/survey_temporal_coverage.csv`
- `reports/tables/survey_field_missingness.csv`
- `reports/figures/survey_trips_by_month.png` when calendar month/date fields exist
- `reports/figures/survey_trips_by_hour.png` when departure-time fields exist

Observed-count outputs:

- `data/raw/observed_counts/mdot_sha_aadt/`
- `data/processed/observed_counts/mdot_sha_aadt/observed_counts_mdot_sha_study_area.geoparquet`
- `data/metadata/observed_counts_mdot_sha_manifest.json`
- `reports/tables/observed_counts_mdot_sha_sample.csv`
- `reports/maps/observed_counts_mdot_sha_coverage.html`
- `reports/maps/observed_counts_mdot_sha_coverage.png` when static plotting succeeds

Tract-geometry outputs:

- `data/raw/census/tiger_tracts/tl_2018_{state}_tract.zip`
- `data/processed/census/tiger_tracts_study_area.geoparquet`
- `data/metadata/census_tract_geometries_manifest.json`

## Known Limitations

- MDOT SHA AADT is annual average daily traffic, not exact trip-day traffic.
- Survey trips may represent person trips, while traffic counts represent vehicles.
- Transit, walk, bike, and other non-roadway vehicle trips should be excluded or handled separately for vehicle-count validation.
- The survey may not cover every month of 2017, 2018, or 2019, so seasonal and annual claims are limited unless the audit proves coverage.
- Spatial matching and route assignment over MDOT AADT segments introduce uncertainty.
- Validation is strongest for auto/vehicle trips and major roads.

## Full Experiment Direction

The package includes interfaces and skeletons for four synthesis methods, four routing methods, validation metrics, and experiment orchestration. The implemented first deliverable is the survey temporal audit plus observed-count acquisition layer.
