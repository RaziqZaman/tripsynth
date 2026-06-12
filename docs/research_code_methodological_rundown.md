# Research Code Methodological Rundown

This document describes the current trip synthesis research harness as implemented in
`src/trip_synth/`, `configs/`, and `scripts/`. The pipeline compares a contrastive
mixed-tabular VAE against non-contrastive and statistical baselines, then evaluates
synthetic trip tables with internal survey-distribution diagnostics and an external
two-prong traffic-count screenline validation.

## 1. Run Entrypoints And Configurations

The main orchestration path is:

1. `scripts/run_pipeline.sh --config <config>` calls `python -m trip_synth.pipeline`.
2. `scripts/run_quick.sh`, `scripts/run_medium.sh`, `scripts/run_paper.sh`, and `scripts/run_full.sh` wrap
   the pipeline with `configs/quick.yaml`, `configs/medium.yaml`,
   `configs/paper.yaml`, and `configs/full.yaml`.
3. `scripts/run_hparam_grid.sh` calls `python -m trip_synth.hparams` with
   `configs/quick.yaml` and `configs/hparam_grid.yaml`.

The run configs define the input CSV, sample size, seed, method list, VAE settings,
baseline settings, validation switches, and geospatial data paths. `quick.yaml` uses
`05sample_transformed_survey.csv`, samples 1,000 synthetic rows, and skips AADT by
default. `medium.yaml` uses `05_transformed_survey.csv`, samples 100,000 rows, and
enables AADT. `full.yaml` samples `population_from_weight_sum`, meaning the requested
synthetic row count is the rounded sum of the survey expansion weights.

Outputs are written under:

```text
outputs/runs/<run_name>/
```

with subdirectories for logs, checkpoints, synthetic samples, metrics, tables,
figures, geospatial intermediates, and cached downloaded data.

## 2. Data Contract And Feature Schema

`configs/schema.yaml` is the authoritative schema for modeling and validation.

The schema treats `weight` as a survey expansion/training weight, not as a generated
trip attribute. `weight` is excluded from all synthetic outputs by
`src/trip_synth/data/postprocessing.py`, but is used for weighted bootstrap sampling,
weighted survey-side validation estimates, and optional weighted VAE training losses.

Categorical features include activities, origin/destination tract FIPS, travel mode,
parking and toll variables, home/work tract fields, demographics, and household or
person statuses. Numeric features include departure time, reported travel time,
household counts, vehicle model year, income category, age, jobs count, and distance.
FIPS fields are cleaned as 11-digit strings where possible. The `year` field is
explicitly treated as vehicle model year, not survey trip year.

The `FittedPreprocessor` learns:

- categorical vocabularies from the training data,
- numeric means, standard deviations, mins, and maxes,
- the final feature column order used for synthesis.

It transforms categorical fields to integer codes, standardizes numeric fields, and
inverse-transforms generated samples back to the original table form.

## 3. Synthesis Methods

The pipeline runs the methods listed in the config. The default comparison set is:

- `weighted_bootstrap`
- `noncontrastive_vae`
- `contrastive_vae`
- `bayesian_network`

For every method, the synthetic table is saved as:

```text
outputs/runs/<run_name>/samples/<method>_synthetic.csv
```

Each synthetic table is checked by `assert_synthetic_contract`: excluded fields such
as `weight` must not appear, and `year` must remain integer typed when present.

### Weighted Bootstrap

`src/trip_synth/baselines/weighted_bootstrap.py` cleans the survey table, computes
sampling probabilities proportional to nonnegative survey weights, and resamples rows
with replacement. This baseline preserves empirical rows and weighted marginal
distributions well, but it is expected to have high exact-row copy rate by design.

### Bayesian Network Baseline

`src/trip_synth/baselines/bayesian_network.py` discretizes numeric columns, estimates
weighted mutual information between features, and builds a maximum-spanning-tree
dependency structure, equivalent to a Chow-Liu-style single-parent network. It samples
from the root marginal and conditional probability tables. Numeric draws are decoded
by sampling observed training values from the sampled bin. Medium, paper, and full configs can cap method generation via `method_sample_caps`;
the pipeline writes a per-method `sample_expansion_factor`, and AADT validation
scales synthetic screenline counts back to the full target population. The medium
config now targets the expanded population while generating 100,000 rows for the
bootstrap/VAE methods and 10,000 rows for the Bayesian network. The paper-scale
config caps every method at 1,000,000 generated rows while retaining the
population-scale validation target.

### Mixed-Tabular VAE

`src/trip_synth/models/tabular_vae.py` defines a VAE for mixed categorical and numeric
tabular data:

- categorical fields are embedded,
- numeric fields are standardized and concatenated with the embeddings,
- an MLP encoder predicts latent `mu` and `logvar`,
- an MLP decoder predicts categorical logits and numeric outputs.

Training in `src/trip_synth/models/train.py` splits the input into train and
validation partitions, fits the preprocessor on the training partition, and optimizes:

```text
loss = reconstruction + beta_kl * KL + lambda_contrastive * InfoNCE
```

The reconstruction term is categorical cross-entropy plus numeric MSE or Huber loss.
The KL term regularizes the latent distribution. If training weights are enabled, row
losses are weighted by transformed survey weights. The default transform caps high
weights at a quantile and normalizes the mean weight to one.

The `noncontrastive_vae` method is an ablation of the same architecture with
`lambda_contrastive = 0`. The `contrastive_vae` adds an InfoNCE representation
regularizer. Positive examples are light perturbations that avoid key OD and timing
columns, while negatives are marginal random draws or shuffled feature values,
depending on config.

Sampling in `src/trip_synth/models/sample.py` draws latent vectors from a standard
normal distribution, decodes them, samples categorical codes from softmax
probabilities with configurable temperature, inverse-transforms numeric values, clips
numeric values to the training range by default, and appends `synthetic_method` and
`synthetic_run_id`.

## 4. Marginal Evaluation

Marginal validation is implemented in `src/trip_synth/validation/marginals.py`.

For each method, the real survey table and synthetic table are compared column by
column using the reference preprocessor's feature list. Survey-side estimates use the
survey `weight` column when available. Synthetic-side estimates are unweighted because
synthetic records are the generated analysis population.

Categorical columns use:

- weighted survey proportions,
- synthetic proportions,
- total variation distance,
- Jensen-Shannon distance,
- top absolute category errors.

Numeric columns use:

- weighted survey mean, standard deviation, and quantiles,
- synthetic mean, standard deviation, and quantiles,
- mean absolute error,
- standard-deviation absolute error,
- one-dimensional Wasserstein distance,
- Kolmogorov-Smirnov statistic.

The method-level summary includes `mean_categorical_tv`, `mean_numeric_ks`, and
`marginal_similarity_score = 1 - mean(cat TV and numeric KS)`. Detailed JSON is saved
under `metrics/<method>_marginals.json`, and plots are saved under
`figures/marginals/<method>/`.

## 5. Cross-Marginal Evaluation

Cross-marginal validation is implemented in
`src/trip_synth/validation/cross_marginals.py`. It adds derived bins for departure
time, reported travel time, distance, and county FIPS. It then compares selected
two-way distributions such as:

- origin activity by destination activity,
- travel mode by departure-time bin,
- origin county by destination county,
- day of week by departure-time bin,
- distance bin by reported-travel-time bin,
- vehicle model year by vehicle.

Each cross-tab uses weighted survey shares and unweighted synthetic shares. Metrics
are total variation distance, Jensen-Shannon distance, and top cell absolute errors.
The method summary includes `mean_cross_tv` and
`cross_marginal_similarity_score = 1 - mean_cross_tv`.

## 6. Privacy And Copying Diagnostics

Privacy diagnostics are implemented in `src/trip_synth/validation/privacy.py`.

The code computes:

- exact-row copy rate against the real feature table,
- duplicate rate inside the synthetic table,
- invalid categorical value rate,
- novel origin-destination pair share,
- a simple `privacy_score = 1 - copy_rate - invalid_category_rate`.

Nearest-neighbor privacy distances are disabled in the active pipeline because they
scale as a synthetic-by-real distance scan and become a bottleneck in medium/full
experiments.

This is not a formal privacy guarantee. It is a diagnostic layer to avoid confusing
good marginal fit with row memorization. The weighted bootstrap baseline should score
poorly on copying because it intentionally resamples observed rows.

## 7. Hyperparameter Search

`src/trip_synth/hparams.py` runs a capped or full grid over VAE hyperparameters. Each
trial trains one configured method, samples up to 2,000 rows, evaluates marginals,
cross-marginals, and privacy, then records:

```text
score = marginal_similarity_score
      + cross_marginal_similarity_score
      + privacy_score
      - invalid_category_rate
      - exact_row_copy_rate
```

Results are saved in `outputs/runs/<run_name>_hparams/tables/hparam_results.csv`,
with supporting figures for score heatmaps and parallel-coordinate views.

## 8. External Two-Prong Traffic Count Validation

External traffic-count validation is implemented across:

- `src/trip_synth/validation/aadt_screenlines.py`
- `src/trip_synth/validation/aadt_validation.py`
- `src/trip_synth/validation/route_assignment.py`

It is enabled in `medium.yaml` and `full.yaml` with `validation.run_aadt: true` and `aadt_validation.validation_mode: two_prong`. The quick config keeps `run_aadt: false`, but carries the same two-prong settings for smoke testing or manual validation.

### External Data Used

The geospatial foundation uses the 2019 Census TIGER tract vintage for Maryland and DC:

- `https://www2.census.gov/geo/tiger/TIGER2019/TRACT/tl_2019_24_tract.zip`
- `https://www2.census.gov/geo/tiger/TIGER2019/TRACT/tl_2019_11_tract.zip`

The MDOT SHA AADT FeatureServer is used for dense station point geometry, station identifiers, and annual-average count fields:

https://mdgeodata.md.gov/imap/rest/services/Transportation/MD_AnnualAverageDailyTraffic/FeatureServer/0?f=pjson

FHWA TMAS monthly continuous count station files supply exact temporal observations for the sparse hourly tier:

https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/

The repository includes:

```text
data/external/mdot_daily_counts/station_daily_counts.csv
data/external/mdot_daily_counts/station_hourly_counts.csv
data/external/mdot_daily_counts/fhwa_tmas_direction_lane_daily_counts_2017-10-01_2019-07-31.csv
data/external/mdot_daily_counts/fhwa_tmas_station_metadata_2017_2019.csv
data/external/mdot_daily_counts/fhwa_tmas_daily_counts_manifest.json
```

The FHWA build currently covers 64 Maryland continuous/permanent stations from October 1, 2017 through July 31, 2019, yielding 37,800 station-day rows and 907,200 station-hour rows. FHWA station IDs such as `0P0009` are normalized to MDOT-style IDs such as `P0009` so they can match the MDOT AADT station map.

VDOT and DDOT public traffic-volume layers were also checked. They are useful for future dense annual-average spatial validation around Virginia/DC, but the public layers found so far expose annual-average line volumes rather than exact station-day or hourly count totals:

- VDOT hub: `https://www.virginiaroads.org/datasets/VDOT::traffic-volume/about`
- VDOT 2018: `https://services.arcgis.com/p5v98VHDX9Atv3l7/arcgis/rest/services/Traffic_Volume_ADT_2018/FeatureServer/0`
- VDOT 2019: `https://services.arcgis.com/p5v98VHDX9Atv3l7/arcgis/rest/services/VDOT_Traffic_Volume_2019a/FeatureServer/0`
- DDOT 2017: `https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Transportation_TrafficVolume_WebMercator/MapServer/134`
- DDOT 2018: `https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Transportation_TrafficVolume_WebMercator/MapServer/154`
- DDOT 2019: `https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Transportation_TrafficVolume_WebMercator/MapServer/167`

### Screenline Construction

The screenline proxy remains a tract-boundary method rather than a full network assignment:

1. Load 2019 MD/DC TIGER tracts and project them to the configured CRS.
2. Build adjacent tract pairs that share a boundary.
3. Convert each adjacent pair into a `screenline_id`.
4. Load MDOT AADT station points.
5. Search for stations near each tract boundary using configured buffers.
6. Apply `station_membership_rule: boundary_closer_than_nearest_tract_centroid`: a station is retained only when it is closer to the shared tract boundary than to the nearer adjacent tract centroid.
7. Sum retained annual-average station counts to form dense observed screenline totals.
8. Save `geo/screenline_station_map.parquet` with annual station count, boundary distance, centroid distance, and membership-rule metadata.

### Dense Annual-Average Tier

The annual tier estimates long-term average daily synthetic screenline crossings. When `tdate_week` and `tdate_dow` are present, synthetic trips are assigned exact reconstructed dates and crossings are averaged across observed synthetic dates. Otherwise, the code falls back to day-of-week averaging.

Annual observed counts are the dense screenline totals built from MDOT AADT/AAWDT station sets. Outputs include:

```text
tables/aadt_annual_validation_summary.csv
tables/aadt_annual_top_residual_screenlines.csv
metrics/aadt_annual_screenline_comparisons.parquet
```

The combined compatibility outputs still exist:

```text
tables/aadt_validation_summary.csv
metrics/aadt_screenline_comparisons.parquet
```

### Sparse Hourly TMAS Tier

The hourly tier compares synthetic hourly screenline crossings to FHWA TMAS station-hour counts. Synthetic crossing time is estimated from:

```text
crossing_time = departure_time_minutes + reported_travel_time * path_fraction
```

where `path_fraction = (path_position + 1) / (number_of_crossings + 1)` along the ordered tract-crossing path.

For observed hourly counts, FHWA TMAS stations are matched to their annual-average station twins by normalized station ID. For a screenline `L`, station `s` has annual-average count `A_s` and the whole retained screenline station set has total `A_L`, so:

```text
p_s = A_s / A_L
observed_screenline_hour = sum(FHWA_hourly_count_s) / sum(p_s)
```

This expands sparse FHWA hourly counts using the dense annual-average station share. It is a scaling assumption, not a direct full-screenline hourly observation.

Hourly outputs include:

```text
tables/aadt_hourly_validation_summary.csv
tables/aadt_hourly_top_residual_screenlines.csv
metrics/observed_hourly_screenline_counts.parquet
metrics/aadt_hourly_screenline_comparisons.parquet
metrics/<method>_hourly_virtual_screenline_counts.parquet
```

### Metrics And Interpretation

Both tiers compute raw/log correlations, RMSE, MAE, guarded MAPE, SMAPE, median APE, bias, R-squared, GEH mean, GEH threshold shares, and log calibration slope/intercept. The annual tier should be interpreted as dense annual-average spatial validation. The hourly tier should be interpreted as sparse FHWA temporal validation scaled by annual station shares. Neither tier is full network traffic assignment; OD paths still use tract representative-point lines and tract-boundary crossings.

## 9. Methodological Bottom Line

The research code has three evaluation tiers:

1. Internal one-way fidelity: per-column marginal distributions.
2. Internal dependence fidelity: selected behaviorally meaningful two-way
   cross-marginals.
3. External traffic plausibility: dense annual-average screenline counts and sparse
   scaled FHWA hourly counts compared with synthetic virtual screenline crossings.

Together, these tiers let the experiment distinguish methods that merely reproduce
individual feature frequencies from methods that preserve joint trip structure and
produce plausible aggregate roadway-volume patterns. The MDOT layer is especially
valuable because it is independent of the survey sample, but its conclusions should
be stated as two-prong screenline validation under a centroid-line proxy, not as
fully routed traffic-count calibration.
