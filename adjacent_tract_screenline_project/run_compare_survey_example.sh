#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
python src/compare_survey_direct_adjacent_od.py \
  --survey-csv ../00__combined-flat-survey.csv \
  --screenlines-csv outputs/tract_pair_aadt_station_screenlines.csv \
  --config config.json \
  --out outputs/tract_pair_direct_adjacent_od_comparison.csv
