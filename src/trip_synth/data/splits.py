from __future__ import annotations

import numpy as np
import pandas as pd


def train_val_split(
    df: pd.DataFrame, val_fraction: float = 0.2, seed: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 5:
        return df.copy(), df.copy()
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(df) * val_fraction)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)
