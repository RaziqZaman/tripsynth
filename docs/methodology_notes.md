# Methodology Notes

1. Survey weights are expansion weights, not ordinary trip features.
2. `weight` is excluded from synthesis and from every synthetic output table.
3. Training with raw weights can overemphasize a small number of large-weight observations, so the VAE config supports normalized, sqrt, log, capped, and capped-normalized transforms.
4. Weighted bootstrap uses weights as sampling probabilities and is expected to have a high copy rate.
5. The default centroid-line tract crossing path is a geometric proxy for route assignment, not true network assignment.
6. The geospatial validation now uses the configured 2019 TIGER tract vintage for the 2017-2019 survey window.
7. The default external validation is two-prong: dense annual-average screenline validation from MDOT AADT/AAWDT station sets plus sparse FHWA TMAS hourly validation scaled by annual station share.
8. `tdate_week` is interpreted as whole weeks after January 1, 2017. With `tdate_dow_encoding: iso_monday_1`, Monday is 1 and Sunday is 7.
9. Contrastive loss is a representation regularizer and must be ablated against an otherwise identical non-contrastive VAE.
10. Privacy and copying diagnostics are necessary because marginal fit alone can reward memorization.
11. Bayesian-network generation is capped in medium/full configs and scaled back up during traffic-count validation with a recorded sample-expansion factor.

## AADT Temporal Alignment

The transformed schema column `year` is vehicle model year, not survey trip year, and must not be used for AADT temporal alignment. The survey trip collection window is October 2017 through July 2019. Two-prong validation reconstructs `trip_date` from `tdate_week` and `tdate_dow`; the annual tier averages synthetic crossings over reconstructed dates, and the hourly tier combines reconstructed dates with `departure_time_minutes` and `reported_travel_time` to estimate screenline crossing hours.
