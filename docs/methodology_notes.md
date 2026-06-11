# Methodology Notes

1. Survey weights are expansion weights, not ordinary trip features.
2. `weight` is excluded from synthesis and from every synthetic output table.
3. Training with raw weights can overemphasize a small number of large-weight observations, so the VAE config supports normalized, sqrt, log, capped, and capped-normalized transforms.
4. Weighted bootstrap uses weights as sampling probabilities and is expected to have a high copy rate.
5. The default centroid-line tract crossing path is a geometric proxy for route assignment, not true network assignment.
6. AADT and AAWDT are average traffic-volume measures, not exact trip-date counts unless daily or hourly station data are available.
7. Public AAWDT is preferred for weekday synthetic trips; otherwise the validation falls back to AADT.
8. Contrastive loss is a representation regularizer and must be ablated against an otherwise identical non-contrastive VAE.
9. Privacy and copying diagnostics are necessary because marginal fit alone can reward memorization.
10. Downstream screenline validation is an aggregate plausibility check, not proof of exact route realism.
