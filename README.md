# Trip Synthesis

Trip Synthesis is a reproducible research-code pipeline for augmenting limited household travel-behavior observations, routing the synthesized trips, and validating the routed roadway volumes against observed traffic-count proxies.

The repository is designed for a TRB-quality experimental workflow, but it starts from a careful premise: the local 2017-2019 household travel survey may not cover every month, day, or time period in those years. The first pipeline step audits the survey's actual temporal coverage before any synthesis, routing, or validation claim is made.

## Baseline Validation

The default validation mode is **spatial roadway-volume proxy validation against AADT**. FHWA HPMS 2018 AADT is an annual-average roadway proxy. It is useful for asking whether a synthesis-routing pipeline reproduces spatial patterns of roadway demand; it is not same-day, same-month, or exact temporal ground truth for every survey trip.

Temporal validation is opt-in and guarded. It only runs when the survey audit and the observed-count dataset both expose overlapping dates or months.

## Required Local Input

Configure the household travel survey in `configs/default.yaml`. The current default points at the local transformed survey:

```yaml
survey:
  path: 00_survey-transformed.csv
```

Schema mapping is explicit. Do not map calendar `survey_year`, `survey_month`, or `trip_date` unless those fields truly describe survey/trip dates. In the provided transformed file, `year` appears to be vehicle model year, so it is intentionally not used as survey year.

## First Commands

Run the survey audit:

```bash
PYTHONPATH=src .venv/bin/python -m tripsynth.cli audit-survey --config configs/default.yaml
```

Fetch public observed counts from FHWA HPMS 2018 for DC, Maryland, and Virginia:

```bash
PYTHONPATH=src .venv/bin/python -m tripsynth.cli fetch-counts --config configs/default.yaml
```

The fetch step downloads ArcGIS FeatureServer data, normalizes to a common observed-count schema, filters to the configured DMV study-area counties, and writes manifests, sample tables, and coverage maps.

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

- `data/raw/observed_counts/fhwa_hpms_2018/{state}/`
- `data/processed/observed_counts/fhwa_hpms_2018/observed_counts_hpms_2018_dmv.geoparquet`
- `data/metadata/observed_counts_manifest.json`
- `reports/tables/observed_counts_manifest.csv`
- `reports/tables/observed_counts_sample.csv`
- `reports/maps/observed_counts_coverage.html`
- `reports/maps/observed_counts_coverage.png` when static plotting succeeds

## Known Limitations

- HPMS AADT is annual average daily traffic, not exact trip-day traffic.
- Survey trips may represent person trips, while traffic counts represent vehicles.
- Transit, walk, bike, and other non-roadway vehicle trips should be excluded or handled separately for vehicle-count validation.
- The survey may not cover every month of 2017, 2018, or 2019, so seasonal and annual claims are limited unless the audit proves coverage.
- Spatial matching from routed links to observed HPMS segments introduces uncertainty.
- Validation is strongest for auto/vehicle trips and major roads.

## Full Experiment Direction

The package includes interfaces and skeletons for four synthesis methods, four routing methods, validation metrics, and experiment orchestration. The implemented first deliverable is the survey temporal audit plus observed-count acquisition layer.
