# Spatial Pattern Tests

These tests focus on whether modeled volumes preserve the spatial pattern of observed traffic counts across count links within each county. `paired_spatial_pattern_tests.csv` uses paired bootstrap confidence intervals and paired permutation p-values for synthetic-minus-real metric differences. Positive differences favor synthetic for all listed pattern metrics. `residual_morans_i_tests.csv` reports Moran's I on residuals; lower absolute values indicate less spatial clustering in model error.
