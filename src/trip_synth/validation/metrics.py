from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def weighted_proportions(values: pd.Series, weights: pd.Series | np.ndarray | None = None) -> pd.Series:
    vals = values.astype("string").fillna("-1")
    if weights is None:
        counts = vals.value_counts(dropna=False)
    else:
        w = pd.to_numeric(pd.Series(weights), errors="coerce").fillna(0.0).clip(lower=0).to_numpy(float)
        tmp = pd.DataFrame({"value": vals.to_numpy(), "weight": w})
        counts = tmp.groupby("value", dropna=False)["weight"].sum()
    total = float(counts.sum())
    if total <= 0:
        return counts.astype(float)
    return (counts / total).sort_index()


def align_distributions(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray, list[str]]:
    keys = sorted(set(a.index.astype(str)) | set(b.index.astype(str)))
    aa = a.reindex(keys, fill_value=0.0).to_numpy(float)
    bb = b.reindex(keys, fill_value=0.0).to_numpy(float)
    return aa, bb, keys


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    return float(math.sqrt(max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m))))


def weighted_numeric_summary(values: pd.Series, weights: pd.Series | np.ndarray | None = None) -> dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "q05": float("nan"), "q50": float("nan"), "q95": float("nan")}
    if weights is None:
        w = np.ones_like(x)
    else:
        w_all = pd.to_numeric(pd.Series(weights), errors="coerce").fillna(0.0).clip(lower=0).to_numpy(float)
        w = w_all[pd.to_numeric(values, errors="coerce").notna().to_numpy()]
        if w.sum() <= 0:
            w = np.ones_like(x)
    mean = float(np.average(x, weights=w))
    var = float(np.average((x - mean) ** 2, weights=w))
    return {
        "mean": mean,
        "std": math.sqrt(max(0.0, var)),
        "q05": float(np.quantile(x, 0.05)),
        "q50": float(np.quantile(x, 0.50)),
        "q95": float(np.quantile(x, 0.95)),
    }


def wasserstein_1d(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").dropna().sort_values().to_numpy(float)
    y = pd.to_numeric(b, errors="coerce").dropna().sort_values().to_numpy(float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    q = np.linspace(0, 1, min(1000, max(10, min(x.size, y.size))))
    return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))


def ks_statistic(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").dropna().sort_values().to_numpy(float)
    y = pd.to_numeric(b, errors="coerce").dropna().sort_values().to_numpy(float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    values = np.sort(np.unique(np.concatenate([x, y])))
    fx = np.searchsorted(x, values, side="right") / x.size
    fy = np.searchsorted(y, values, side="right") / y.size
    return float(np.max(np.abs(fx - fy)))


def geh(observed: np.ndarray, synthetic: np.ndarray) -> np.ndarray:
    observed = np.asarray(observed, dtype=float)
    synthetic = np.asarray(synthetic, dtype=float)
    denom = np.maximum((observed + synthetic) / 2.0, 1e-9)
    return np.sqrt(2.0 * (synthetic - observed) ** 2 / denom)


def regression_slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    return float(slope), float(intercept)


def safe_corr(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if method == "spearman":
        x = pd.Series(x).rank().to_numpy(float)
        y = pd.Series(y).rank().to_numpy(float)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def numeric_to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
