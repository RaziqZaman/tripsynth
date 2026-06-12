# Run Summary: paper_wctr_1m_vae_sweep

Runtime seconds: 2660.4

## Methods
- weighted_bootstrap: 1000000 synthetic rows
- bayesian_network: 1000000 synthetic rows
- noncontrastive_vae: 1000000 synthetic rows
- contrastive_vae: 1000000 synthetic rows

## Best Methods
- Best marginal method: weighted_bootstrap
- Best cross-marginal method: weighted_bootstrap
- Best privacy/non-copying method: bayesian_network
- Best AADT method: see tables/aadt_validation_summary.csv
- Contrastive VAE vs non-contrastive VAE: marginal TV delta = 0.0031 (negative favors contrastive)

## Warnings
- none

## Key Figure Paths
- figures/poster/01_architecture_contrastive_vae.png
- figures/poster/04_marginal_distance_leaderboard.png
- figures/poster/05_cross_marginal_error_heatmap.png
- figures/poster/06_privacy_copy_rate_by_method.png
- figures/poster/11_best_method_summary_panel.png

## AADT Status
- {'status': 'ok', 'validation_mode': 'two_prong', 'tiers': {'annual_average': {'status': 'ok', 'screenlines_compared': 3107, 'comparison_basis': 'average_day'}, 'hourly_tmas_scaled': {'status': 'ok', 'screenlines_compared': 56, 'screenline_hours_compared': 642456, 'hourly_counts': {'file': 'data/external/mdot_daily_counts/station_hourly_counts.csv', 'station_id_column': 'station_id', 'date_column': 'date', 'hour_column': 'hour', 'count_column': 'observed_count'}, 'scaling_method': 'FHWA station hourly count divided by covered annual station share of the screenline'}}, 'screenlines_compared': 3107, 'methods': ['bayesian_network', 'contrastive_vae', 'noncontrastive_vae', 'weighted_bootstrap'], 'temporal_framing': 'Two-prong annual-average plus FHWA TMAS hourly validation.', 'trip_week_origin_date': '2017-01-01', 'tdate_dow_encoding': 'iso_monday_1', 'daily_counts': None, 'hourly_counts': {'file': 'data/external/mdot_daily_counts/station_hourly_counts.csv', 'station_id_column': 'station_id', 'date_column': 'date', 'hour_column': 'hour', 'count_column': 'observed_count'}, 'survey_period': {'start_date': '2017-10-01', 'end_date': '2019-07-31', 'trip_year_column': None, 'note': 'The schema column `year` is vehicle model year, not survey trip year.'}, 'synthetic_population_basis': 'average_day', 'expand_date_strata_to_average_day': True, 'assignment_note': 'OD-to-screenline paths use centroid-line tract crossings as a geometric proxy, not true route assignment.', 'screenlines': {'status': 'ok', 'tracts': 2232, 'adjacent_pairs': 5927, 'candidate_screenlines': 3257, 'screenlines': 3257, 'screenlines_with_stations': 3107, 'stations_mapped': 6725, 'observed_count_field': 'AAWDT_2018', 'observed_count_temporal_label': 'Annual-average counts plus FHWA TMAS hourly counts', 'comparison_basis': 'two_prong', 'station_membership_rule': 'boundary_closer_than_nearest_tract_centroid', 'boundary_distance_ratio_threshold': 0.5, 'screenline_candidate_scope': 'observed_station_buffer', 'survey_period': {'start_date': '2017-10-01', 'end_date': '2019-07-31', 'trip_year_column': None, 'note': 'The schema column `year` is vehicle model year, not survey trip year.'}, 'segments_note': 'MDOT segment layer was optional; current downloaded layer may be empty.', 'method_note': 'Screenline assignment is a geometric proxy, not true route assignment.'}}
