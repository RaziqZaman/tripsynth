#!/usr/bin/env bash
set -euo pipefail

REAL_EVENTS="${REAL_EVENTS:-06x_stage1_md_internal/matsim_real/ITERS/it.0/0.events.xml.gz}"
SYNTHETIC_EVENTS="${SYNTHETIC_EVENTS:-06x_stage1_md_internal/matsim_synthetic/ITERS/it.0/0.events.xml.gz}"

.venv/bin/python 06_extract-matsim-link-volumes.py \
  --events "$REAL_EVENTS" \
  --out 06x_stage1_md_internal/real_matsim_link_volumes.csv

.venv/bin/python 06_extract-matsim-link-volumes.py \
  --events "$SYNTHETIC_EVENTS" \
  --out 06x_stage1_md_internal/synthetic_matsim_link_volumes.csv

.venv/bin/python 06_aggregate-matsim-volumes.py \
  --link-volumes 06x_stage1_md_internal/real_matsim_link_volumes.csv \
  --out 06x_stage1_md_internal/real_assigned_link_counts.csv

.venv/bin/python 06_aggregate-matsim-volumes.py \
  --link-volumes 06x_stage1_md_internal/synthetic_matsim_link_volumes.csv \
  --out 06x_stage1_md_internal/synthetic_assigned_link_counts.csv

.venv/bin/python 06_compare-stage1-counts.py \
  --scenario-totals 06x_stage1_md_internal/scenario_totals.csv \
  --observed-counts 06x_stage1/observed_counts.csv \
  --real-assigned 06x_stage1_md_internal/real_assigned_link_counts.csv \
  --synthetic-assigned 06x_stage1_md_internal/synthetic_assigned_link_counts.csv \
  --out 06x_stage1_md_internal/count_validation.csv \
  --summary-out 06x_stage1_md_internal/count_validation_summary.csv
