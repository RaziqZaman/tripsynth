#!/usr/bin/env python3
"""Deterministic household-level train/holdout split helpers."""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple


def household_key(row: Dict[str, object], fallback_index: int) -> str:
    raw = row.get("household_id", "")
    key = "" if raw is None else str(raw).strip()
    if key:
        return key
    return f"__row_{fallback_index:09d}"


def split_rows_by_household(
    rows: Sequence[Dict[str, object]],
    holdout_fraction: float,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    indexed = [(household_key(row, i), row) for i, row in enumerate(rows)]
    keys = sorted({key for key, _ in indexed})

    rng = random.Random(seed)
    rng.shuffle(keys)

    holdout_n = int(round(len(keys) * holdout_fraction))
    holdout_n = max(1, min(len(keys) - 1, holdout_n)) if len(keys) > 1 else len(keys)
    holdout_keys = set(keys[:holdout_n])

    train_rows: List[Dict[str, object]] = []
    holdout_rows: List[Dict[str, object]] = []
    for key, row in indexed:
        if key in holdout_keys:
            holdout_rows.append(dict(row))
        else:
            train_rows.append(dict(row))
    return train_rows, holdout_rows
