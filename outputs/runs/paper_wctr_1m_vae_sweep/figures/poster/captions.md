# Poster Figure Captions

Figures summarize the synthesis pipeline, validation metrics, privacy diagnostics, and optional two-prong screenline validation.
Traffic-count figures are placeholders in quick runs because external geospatial validation is disabled by configuration.

Figure 12 compares equal-sized samples of weighted-bootstrap and contrastive-VAE trips as origin-destination tract lines. It is a tract-centroid visualization for coverage, not route assignment.
Figure 13 zooms to one origin tract inside an auto-selected 60-tract Maryland zone and shows which nearby destination tracts are reached by each method.
Figure 14 maps MDOT stations assigned to annual AADT screenlines. Point color is the contrastive-VAE synthetic-to-observed screenline ratio, and point size is station AAWDT.
Figure 15 zooms to one OD-pair example and traces the station-matched screenlines used to compare synthetic virtual crossings with observed AAWDT counts.
Figure 16 repeats the OD-pair screenline view for an example where the contrastive VAE has the lowest path-level annual-average count error among all compared synthetic methods.
Figure 17 shows a single-TMAS-station hourly validation profile where the contrastive VAE has the lowest 24-hour count-profile error among compared synthetic methods.
Figure 18 repeats the hourly validation profile as an aggregate over all measured TMAS screenlines using common comparable hourly cells across methods.
Figure 19 summarizes hourly TMAS traffic-count RMSE reductions relative to the survey-sampled baseline for all non-baseline synthesis methods.
Figure 20 pairs hourly TMAS RMSE reductions with annual screenline first-place counts, where contrastive VAE wins the most annual screenlines by absolute count error.
