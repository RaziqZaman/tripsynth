# Evidence Manifest

Audit date: 2026-07-30  
Repository branch and commit audited: `wctr-aadt`, `3363e21ed781af9c5d659c3fed6d08ea661170e2`  
Frozen experiment: `outputs/runs/paper_wctr_1m_vae_sweep`  
Manuscript analysis entry point: `trb/scripts/generate_manuscript_assets.py`

## Interpretation of this file

This manifest traces the important empirical and methodological claims in `trb/trb_template.tex` to repository evidence. “Reproduced” means recalculated during this manuscript task from retained data or predictions. “Verified” means checked against a frozen artifact, configuration, log, or implementation but not rerun end to end. Existing prose and the user prompt were not treated as numerical evidence.

The evidence hierarchy was: (1) calculations reproduced from repository data; (2) saved outputs produced by documented code; (3) logs, configuration, and implementation; (4) historical repository files; and (5) prior prose and the prompt. No raw research data were modified. New calculations and machine-readable outputs are confined to `trb/scripts/`, `trb/tables/`, and `trb/generated/`.

## A. Analytical data and synthesis task

| ID | Manuscript claim | Supporting evidence | Definition, unit, denominator, and status |
|---|---|---|---|
| A1 | The analytical table contains 110,926 retained records. | `05_transformed_survey.csv`; `trb/tables/survey_data_summary.csv`; `trb/scripts/generate_manuscript_assets.py:1208-1229`. | Rows in the final transformed CSV. **Reproduced.** Earlier stages contain 163,290 rows (`01x_distilled_survey.csv`, `02_filtered_survey.csv`) and 110,926 rows (`03_only_vehicle_trips_survey.csv`, `04_null_filled_survey.csv`). |
| A2 | A record is conservatively a “retained vehicle-mode-coded trip row.” | Stage-2/stage-3 CSV difference and historical preprocessing. | The filter retains `travel_mode` codes 3, 4, or 5 and nonmissing `vehicle_occupancy`. **Reproduced.** No current dictionary establishes those mode labels or the original trip definition; both need author verification. |
| A3 | There are 44 generated fields plus one excluded weight: 26 categorical, 17 ordinal-integer, and one continuous. | `configs/schema.yaml`; `src/trip_synth/data/schema.py:100-102`; `05_transformed_survey.csv`; `trb/tables/survey_data_summary.csv`. | Schema column types. **Verified/reproduced.** |
| A4 | The retained weight totals 19,287,211.907 units and is not generated. | `05_transformed_survey.csv`; `configs/schema.yaml:1-6`; `src/trip_synth/data/schema.py:100-102`; `trb/generated_results.tex`. | Sum of positive `weight`; range 9–2,554. **Reproduced.** Stage-5 `weight` equals stage-4 `wthhfin` row for row. Historical `00_consolidate-survey.py` at commit `fad425c6` removed `wttrdfin` and `wtperfin` but retained `wthhfin`; the paper therefore calls it a household-final weight used at trip-row level, not a verified trip-final weight. |
| A5 | The table spans 423 reconstructed date strata from 2017-10-10 through 2019-07-11 and weekdays only. | `tdate_week`/`tdate_dow` in `05_transformed_survey.csv`; `trb/scripts/generate_manuscript_assets.py:1210-1223`. | Distinct reconstructed week/day combinations. **Reproduced.** The support is narrower than the configured 2017-10-01–2019-07-31 period. |
| A6 | There are 110,824 valid-OD rows and 58,897 valid directed tract pairs. | `05_transformed_survey.csv`; `trb/tables/survey_data_summary.csv`; `trb/scripts/generate_manuscript_assets.py:1215-1223`. | Rows and ordered pairs after invalid-code exclusion. **Reproduced.** There are 58,925 pairs before valid-code filtering. |
| A7 | Eleven reported-time values and one distance value use `-1` sentinels without separate indicators. | `05_transformed_survey.csv`; `src/trip_synth/data/preprocessing.py:59-103`. | Row counts in the two numeric fields. **Reproduced.** They enter numeric preprocessing and are not described as valid measurements. |
| A8 | Median distance is 4.102 miles (IQR 1.640–10.163); median reported time is 15 minutes (IQR 10–30). | `05_transformed_survey.csv`; `trb/tables/survey_data_summary.csv`. | Unweighted row-level statistics with implemented sentinel handling. **Reproduced.** |
| A9 | VAE development uses an 80/20 row split: 88,741 training and 22,185 validation rows. | `src/trip_synth/data/splits.py:7-18`; `src/trip_synth/models/train.py:173-184`; frozen metadata. | Random rows, seed 20260611. **Verified.** Identifiers were removed, so household leakage cannot be audited. Bootstrap and Bayesian network fit all 110,926 rows. |
| A10 | Each generated method produced one million rows and used factor 19.287212. | `src/trip_synth/pipeline.py:89-102`; per-method scaling JSON; `trb/tables/internal_validation_results.csv`. | Weight total divided by one million rows. **Verified.** The four synthetic CSVs are absent, so exact content is not bitwise reproducible without resampling/retraining. |

## B. Synthesis methods and model implementation

| ID | Manuscript claim | Supporting evidence | Definition, unit, denominator, and status |
|---|---|---|---|
| B1 | The comparison includes direct weighted expansion, weighted bootstrap, a Chow–Liu Bayesian network, noncontrastive VAE, and matched contrastive VAE. | `src/trip_synth/methods/`; `src/trip_synth/models/`; frozen outputs; direct reconstruction at `trb/scripts/generate_manuscript_assets.py:518-545`. | Direct is the nonsynthesis benchmark; four methods generate samples. **Verified/reproduced.** Bootstrap is not mislabeled as direct survey use. |
| B2 | Bootstrap draws proportional to retained weights; the Bayesian network uses weighted Chow–Liu fitting, at most 16 bins, and smoothing 1.0. | `src/trip_synth/methods/weighted_bootstrap.py:22-28,43-54`; `src/trip_synth/methods/bayesian_network.py:95-140`; `configs/paper.yaml`. | Implemented probabilities, mutual information, and CPT settings. **Verified.** |
| B3 | VAE input is 280 dimensions (26 categorical embeddings and 18 standardized numerics), latent dimension 512, hidden widths 512/499/487/475, ReLU, LayerNorm, dropout 0.02, mirrored decoder. | Frozen preprocess JSON/checkpoints; `configs/paper.yaml:1-43`; model code. | Model dimensions; 8,302,216 parameters per VAE. **Verified.** |
| B4 | AdamW used learning rate 0.0005, batch 2,048, gradient clipping 5, KL coefficient 0.03, epoch cap 1,260, sample temperature 0.75. | `configs/paper.yaml:1-43`; training code/logs. | Frozen run settings. **Verified.** Dependencies/hardware are not frozen and deterministic Torch algorithms were not enabled. |
| B5 | The ablation changes contrast coefficient 0 versus 0.25. | `configs/paper.yaml`; frozen configurations. | Coefficient on encoder contrastive loss. **Verified.** |
| B6 | The contrastive loss is two-logit positive-versus-one-negative cosine cross-entropy at temperature 0.10, not conventional many-negative InfoNCE. | `src/trip_synth/models/losses.py:40-55`; call at `train.py:247-260`. | Unweighted mean. **Verified from code.** |
| B7 | Positive views protect OD tract, week, weekday, and departure time; negatives draw categorical codes uniformly and shuffle each numeric column. | `src/trip_synth/models/train.py:75-100`; `configs/paper.yaml`. | Implemented corruption. **Verified.** “Marginal” mode does not sample empirical categorical marginals. |
| B8 | Selected checkpoints are epochs 195 (noncontrastive) and 196 (contrastive). | Frozen checkpoints, histories, and `outputs/runs/paper_wctr_1m_vae_sweep/logs/pipeline.log`. | Lowest stored validation reconstruction-plus-KL loss; contrastive loss is excluded from checkpoint score. **Verified.** Both ran through epoch 1,260. |
| B9 | Frozen runtimes are 6.84 s bootstrap, 76.44 s Bayesian network, 537.87 s noncontrastive VAE, and 896.31 s contrastive VAE. | `outputs/runs/paper_wctr_1m_vae_sweep/metrics/method_runtime_seconds.json`. | Wall seconds from the second run. **Verified.** Hardware is undocumented, so these are descriptive only. |
| B10 | Traffic counts are not read by implemented training, loss, checkpoint, or hyperparameter-score code. | `src/trip_synth/models/train.py`; `src/trip_synth/models/losses.py`; `src/trip_synth/hparams.py:78-145`; code-path audit. | Code-level information separation. **Verified.** Search outputs and complete human selection history are absent; the paper says “independent diagnostic,” not an unqualified prospective out-of-sample test. |

## C. External data and screenline construction

| ID | Manuscript claim | Supporting evidence | Definition, unit, denominator, and status |
|---|---|---|---|
| C1 | Annual observations are 2018 MDOT SHA AAWDT. | `data/external/mdot_aadt/aadt_points.geojson` (`AAWDT_2018`); frozen config; `trb/tables/traffic_count_summary.csv`. | Vehicles per average weekday at a station. **Verified.** Config says `average_day`, but the paper uses the field’s weekday definition. |
| C2 | Hourly data contain 907,200 station-hours for 64 FHWA TMAS stations over 665 dates, October 2017–July 2019. | `data/external/mdot_daily_counts/station_hourly_counts.csv`; its `README.md`; downloader/parser; `trb/tables/traffic_count_summary.csv`. | Station-hour rows; 99999 is missing. **Reproduced/verified.** Direction/lane records are summed; duplicate submissions retain the 24-hour record with largest daily total. |
| C3 | Annual support is 3,107 positive-count tract-boundary screenlines. | Frozen annual comparison parquet; `trb/tables/annual_validation_results.csv`. | Adjacent 2019 Census-tract boundaries. **Reproduced.** |
| C4 | Hourly support is 56 boundaries mapped to 37 stations; 12 stations are reused. | Frozen station map; `trb/scripts/generate_manuscript_assets.py:549-564`; `trb/tables/traffic_count_summary.csv`. | One station per evaluated hourly boundary. **Reproduced.** Reuse motivates station clustering. |
| C5 | A station is linked to a boundary when boundary distance is less than half its distance to the nearer representative point. | `src/trip_synth/validation/aadt_screenlines.py:313-325,423-520`. | Dimensionless proximity rule. **Verified.** Annual mapping reuses 1,271 stations. |
| C6 | Paths are straight representative-point segments with intersections ordered along the segment. | `src/trip_synth/validation/aadt_validation.py:121-201`; frozen `geo/od_screenline_paths.parquet`; Figure 3. | Geometric proxy. **Verified.** It is not road routing, assignment, or a feasible traveler path. |
| C7 | Crossing time is departure plus reported duration times boundary-position fraction; next-day offsets are retained. | Production `src/trip_synth/validation/aadt_validation.py:496-534`; reproduction `trb/scripts/generate_manuscript_assets.py:536-544,599-607`. | Boundary crossing hour/date. **Reproduced.** Fifty-three survey trips produce 249 next-day crossings; omitting offsets changes direct exact RMSE from 13,553.17 to 13,554.77. |
| C8 | Candidate OD paths are capped at 250,000 after method-ordered concatenation. | `src/trip_synth/validation/aadt_validation.py:96-118`; `configs/paper.yaml:73-75`. | Order: survey, bootstrap, BN, noncontrastive VAE, contrastive VAE. **Verified.** The cap can favor earlier methods’ novel support. |
| C9 | Hourly observed screenline counts scale a mapped station by its annual screenline share. | `src/trip_synth/validation/aadt_validation.py:537-580`. | Hourly station count divided by annual share. **Verified.** It is an extrapolated screenline estimate, not a direct full-boundary measurement. |

## D. Internal fidelity, support, and copying

| ID | Claim/result | Evidence and exact interpretation |
|---|---|---|
| D1 | Table 8 internal metrics. | Frozen summaries copied by `trb/scripts/generate_manuscript_assets.py:960-964` to `trb/tables/internal_validation_results.csv`. Means span 26 categorical fields, 18 numeric fields, and nine cross-tables; support/copy values are row shares of one-million-row samples. Bootstrap: categorical TV 0.003302, numeric KS 0.043884, cross-TV 0.004998, exact copy 1.0, duplicates 0.891771, unseen-OD rows 0. Bayesian network: 0.003368, 0.043685, 0.090114, 0, 0, 0.333040. Noncontrastive VAE: 0.062143, 0.126125, 0.255585, 0, 0, 0.685630. Contrastive VAE: 0.065207, 0.110329, 0.226547, 0, 0, 0.685171. **Reproduced from frozen metrics.** |
| D2 | Contrastive versus ablation: numeric KS −12.5%, cross-TV −11.4%, categorical TV +4.9%. | `trb/tables/internal_validation_results.csv`; `trb/scripts/generate_manuscript_assets.py:395-429`. Lower is better. **Reproduced.** Both VAE internal distances remain worse than BN/bootstrap. |
| D3 | Seven of nine cross-tables improve; two-sided paired Wilcoxon p=0.02734375. Numeric/categorical paired tests are not significant. | Frozen per-field JSON; `trb/scripts/generate_manuscript_assets.py:966-1009`; `trb/tables/internal_paired_tests.csv`. **Reproduced.** Dependent heterogeneous endpoints, no preregistration or multiplicity adjustment; descriptive only. |
| D4 | No exact 44-field match was detected for BN or either VAE. | `src/trip_synth/validation/privacy.py:19-63`; frozen privacy summary. **Verified.** Exact equality is weak with a continuous field and does not establish anonymity, disclosure safety, membership privacy, or differential privacy. |
| D5 | Novel OD share is the share of generated rows on an unseen directed pair. | Privacy implementation and summary. **Verified.** It is not the number/share of distinct feasible new pairs; absent synthetic CSVs prevent recomputing prior unique-pair claims, which are omitted. |

## E. Hourly traffic-count evidence

| ID | Claim/result | Evidence and exact interpretation |
|---|---|---|
| E1 | Archived leaderboard: bootstrap RMSE 14,143.09 and contrastive RMSE 6,944.47, apparent reduction 50.8985%. | Frozen `tables/aadt_hourly_validation_summary.csv`; `trb/scripts/generate_manuscript_assets.py:953-958`. **Reproduced but rejected as the estimate** because methods use different dates/cells and factors 440/511/474/492. Reported only as an audit trigger. |
| E2 | Fair exact-date support: 521,424 cells, 56 boundaries, 37 stations, 427 dates, 24 hours, shared factor 423. | `trb/scripts/generate_manuscript_assets.py:567-631`; `trb/generated/exact_date_common_support_predictions.csv`; `trb/tables/exact_date_common_support_results.csv`. Common boundary–date–hour keys; missing predictions zero. **Reproduced.** |
| E3 | Exact RMSE: direct 13,553.1735; bootstrap 13,880.9143; BN 7,038.4726; noncontrastive VAE 6,909.7027; contrastive VAE 6,919.8886. | `trb/tables/exact_date_common_support_results.csv`; `trb/scripts/generate_manuscript_assets.py:614-631`. RMSE in vehicles per common cell, n=521,424. **Reproduced.** |
| E4 | Contrastive exact RMSE is 48.9427% below direct; 95% CI 42.0811%–56.6092%. | `trb/tables/exact_date_clustered_contrasts.csv`; station resampling at `trb/scripts/generate_manuscript_assets.py:633-671`. Paired 37-station bootstrap, 10,000 replicates, seed 20260730. **Reproduced.** This is the qualified principal result. |
| E5 | Contrastive minus noncontrastive exact RMSE is +10.1859 vehicles; CI −86.7646 to 79.0992. | `trb/tables/exact_date_clustered_contrasts.csv`. **Reproduced.** The large direct contrast cannot be attributed to contrastive regularization. |
| E6 | One generated crossing represents at least about 8.2 thousand vehicles under shared exact-date expansion. | 19.287212 × 423 = 8,158.49; archived method-specific quanta 8,486–9,856. **Reproduced.** Supports interpretation as sparse-cell smoothing rather than calibrated magnitude. |
| E7 | Date-pooled analysis uses a common 56×24=1,344 boundary-hour grid. | `trb/scripts/generate_manuscript_assets.py:738-778`; `trb/tables/hourly_validation_results.csv`. Observed weekdays averaged by boundary/hour; method factors undone; generated crossings summed over dates. **Reproduced.** |
| E8 | Pooled RMSE: direct 6,083.944; bootstrap 6,085.613; BN 5,916.489; noncontrastive 5,929.232; contrastive 6,192.574. | `trb/tables/hourly_validation_results.csv`. Vehicles per boundary-hour, n=1,344. **Reproduced.** The exact-date advantage disappears; BN is best. |
| E9 | Contrastive minus direct pooled RMSE +108.630 (CI −29.741, 257.277); contrastive minus noncontrastive +263.342 (CI 195.647, 321.529). | `trb/tables/paired_hourly_effects.csv`; station clustering at `trb/scripts/generate_manuscript_assets.py:676-735`. 37 clusters, 10,000 replicates. **Reproduced.** |
| E10 | Profile TV: direct 0.302657; bootstrap 0.309302; BN 0.140795; noncontrastive 0.249352; contrastive 0.196686. | `trb/tables/hourly_validation_results.csv`; metric `trb/scripts/generate_manuscript_assets.py:797-808`. Half-L1 distance between normalized 24-hour shares, mean of 56 boundaries. **Reproduced.** |
| E11 | Contrastive profile TV is 21.1214% below ablation (CI 16.6138%–25.7223%) and 35.0136% below direct (25.3218%–44.2798%); BN remains best. | `trb/tables/paired_hourly_effects.csv`. 37-station paired bootstrap. **Reproduced.** Shape benefit, not volume calibration. |

## F. Annual traffic-count evidence

| ID | Claim/result | Evidence and exact interpretation |
|---|---|---|
| F1 | All five alternatives use 3,107 common annual boundaries. | `trb/scripts/generate_manuscript_assets.py:781-794`; frozen annual parquet; `trb/tables/annual_validation_results.csv`. One AAWDT value per boundary. **Reproduced.** |
| F2 | Annual RMSE: direct 56,684; bootstrap 56,659; BN 55,772; noncontrastive 56,145; contrastive 57,843. | `trb/tables/annual_validation_results.csv`; `trb/scripts/generate_manuscript_assets.py:896-945`. Vehicles/boundary, n=3,107. **Reproduced.** BN best, contrastive worst; all methods underpredict. |
| F3 | MAE 29,855–30,789; log correlation 0.169–0.217; contrastive bias −22,119.8 vehicles. | `trb/tables/annual_validation_results.csv`. **Reproduced.** |
| F4 | Unique five-way wins: direct 342, bootstrap 359, BN 772, noncontrastive 808, contrastive 825; one five-way zero tie excluded. | `trb/scripts/generate_manuscript_assets.py:896-945`; `trb/tables/annual_validation_results.csv`; Figure 6. A unique win is strictly minimum absolute error. **Reproduced.** Descriptive ranks, not aggregate superiority. |
| F5 | Contrastive’s 825 wins are 26.6%; 426 occur in observed-volume quartile 1 and 14 in quartile 4; winning boundaries contain 7.46% of observed volume. | Frozen annual parquet plus direct reconstruction and the same unique-win logic. **Reproduced.** Explains conflict between wins and RMSE. |
| F6 | Contrastive is largest-error method on 1,228 boundaries and has worst mean five-way rank, 3.278. | Same annual absolute-error matrix. **Reproduced.** |
| F7 | Annual scaling has a small method-dependent denominator mismatch. | `src/trip_synth/validation/aadt_validation.py:482-491,693-717`; saved count quanta. Routed-date denominators 423/451/450/448 versus stored multipliers 423/452/452/453. **Verified/reproduced.** Correcting it slightly worsens contrastive RMSE and does not alter conclusions. |

## G. Literature and novelty evidence

All 49 entries in `trb/trb_template.bib` were checked against a DOI record, publisher page, original paper/proceedings page, or official institutional source. DOI and URL fields are retained where available. The compiled bibliography is alphabetical and contains exactly the 49 cited keys.

| Literature-dependent claim | Principal cited evidence | Disposition |
|---|---|---|
| Traditional synthesis includes reweighting, combinatorial reconstruction, simulation, and graphical models. | Beckman et al. (1996); Farooq et al. (2013); Sun and Erath (2015); Saadi et al. (2016). | Supported; synthesized rather than presented as an exhaustive history. |
| Population-first workflows differ from direct trip-row synthesis. | Joubert and de Waal (2020); Hörl and Balac (2021); Borysov et al. (2019); Badu-Marfo et al. (2022). | Supported; population-first work is not mischaracterized as inferior. |
| Direct survey-trip, diary, and smart-card precedents exist. | Greaves and Stopher (2000); Kieu et al. (2023); Kressner (2017). | Supported; no “first trip generator” claim. |
| Contrastive/diffusion synthesis precedents exist. | van den Oord et al. (2018); Bahri et al. (2022); Lee et al. (2023); Kotelnikov et al. (2023); Li et al. (2026). | Supported; novelty is narrowed to matched trip-row ablation and external diagnostic combination. |
| Count-based external validation is not unprecedented. | Kressner (2017); Chitturi et al. (2014); Yin et al. (2018); Vo et al. (2026); FHWA (2010). | Supported; explicitly not claimed as the first count validation. |
| Internal fidelity and task-specific/external utility are distinct. | Snoke et al. (2018). | Supported and used to structure validation. |
| The integrated combination appears less common. | Targeted review across 49 verified sources and Tables 1–4. | Qualified as “to our knowledge after a targeted review,” not a universal priority claim. |

`li2026syncmove` was online-first on 2026-07-02 and had no assigned volume/pages at audit time; `oord2018cpc` is an arXiv preprint. Standard `chicago.bst` therefore warns about missing pagination. Those fields were not fabricated.

## H. Reproducibility and immutable evidence

| Artifact | SHA-256 |
|---|---|
| `05_transformed_survey.csv` | `95299cd928184599e521003840f41864ed0fc5c449ee1974252832293535afe9` |
| `data/external/mdot_daily_counts/station_hourly_counts.csv` | `15fd4ad2d13351fd0cf09f09614c74122ff1de73dcb130cfb014510c2363cac6` |
| `data/external/mdot_daily_counts/station_daily_counts.csv` | `278cbf4a060703fec579d9fb58c00f26f47de44787d24fcdbd11d38045bf2361` |
| `data/external/mdot_aadt/aadt_points.geojson` | `802405a50a7e60263e0c52057ac23dbe34c936385069e01201425b3589c17b7c` |
| Frozen hourly comparison parquet | `94f4d424d8c2bc2169153948233ea5693d3f159fd5f9689f70afd467c8e5f772` |
| Frozen annual comparison parquet | `8e43cc7ff982405011c5d4d21757d8a0db768c8ba0051859aeb400a5197ae53b` |
| Frozen OD-screenline path parquet | `a3b9e1a51f2a9c9706df7fc60998fe2e3bd31dce8ffc013fc99649a71de4e653` |

Rebuild manuscript values, tables, machine-readable figure data, and six figures from repository root:

```bash
.venv/bin/python trb/scripts/generate_manuscript_assets.py
```

The script fixes bootstrap seed 20260730 and reads retained survey/frozen experiment artifacts; it does not retrain a generator. All 19 repository tests pass. They cover core contracts, date conversion, crossing timing, station scaling, and metric presence, but not an end-to-end test of common support, expansion algebra, station reuse, the OD cap, or the headline result.

## I. Unresolved evidence and prohibited interpretations

The following remain unresolved rather than inferred: source dictionaries and definitive trip/mode definitions; operational use of the household-final weight; several raw-to-model programs; archived hyperparameter-search results and complete human selection history; household/person separation in the VAE split; frozen dependencies/hardware; one-million-row synthetic CSVs and prior distinct generated-OD counts; road routing/assignment; formal privacy; and generalization beyond this region, seed, survey lineage, and frozen experiment.

Accordingly, the manuscript does **not** interpret 48.9% as a general improvement in calibrated flow, call contrastive VAE the strongest overall external method, equate annual wins with aggregate superiority, or claim formal privacy.
