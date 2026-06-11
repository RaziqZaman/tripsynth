# Trip Synth WCTR Experiment

This repository contains a reproducible research harness for comparing a contrastive-loss mixed-tabular VAE against non-contrastive VAE, weighted bootstrap, Gibbs, and Bayesian-network-style trip synthesis baselines.

The default quick run uses `05sample_transformed_survey.csv`, excludes `weight` from all synthetic outputs, keeps `year` as an integer feature, and skips AADT validation unless external geospatial data are available and enabled.

## Quick Start

```bash
bash scripts/run_quick.sh
```

Outputs are written to:

```text
outputs/runs/<run_name>/
```

The longer runs are:

```bash
tmux new -s trip_quick "bash scripts/run_quick.sh"
tmux new -s trip_medium "bash scripts/run_medium.sh"
tmux new -s trip_full "bash scripts/run_full.sh"
tmux new -s trip_hparams "bash scripts/run_hparam_grid.sh"
```

## Data Notes

Survey `weight` is an expansion/training weight, not a synthesizable trip attribute. It is used for weighted bootstrap probabilities, weighted survey validation estimates, and optional weighted VAE training losses, but it is removed from generated synthetic tables.

AADT/AAWDT validation is framed as an aggregate average-day or average-weekday plausibility check. The default OD-to-screenline path method is a centroid-line tract-crossing proxy rather than true route assignment.
