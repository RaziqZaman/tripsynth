"""Traffic-volume validation metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression


def _arrays(predicted, observed) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(predicted, dtype=float)
    obs = np.asarray(observed, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(obs)
    return pred[mask], obs[mask]


def weighted_mape(predicted, observed, weights=None, epsilon: float = 1.0) -> float:
    pred, obs = _arrays(predicted, observed)
    if pred.size == 0:
        return math.nan
    denom = np.maximum(np.abs(obs), epsilon)
    values = np.abs(pred - obs) / denom
    if weights is None:
        weights_arr = np.maximum(obs, epsilon)
    else:
        weights_arr = np.asarray(weights, dtype=float)
        weights_arr = weights_arr[np.isfinite(weights_arr)]
        if weights_arr.size != values.size:
            weights_arr = np.maximum(obs, epsilon)
    return float(np.average(values, weights=weights_arr))


def rmsle(predicted, observed) -> float:
    pred, obs = _arrays(predicted, observed)
    if pred.size == 0:
        return math.nan
    pred = np.maximum(pred, 0)
    obs = np.maximum(obs, 0)
    return float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(obs)) ** 2)))


def geh_values(predicted, observed) -> np.ndarray:
    pred, obs = _arrays(predicted, observed)
    denom = pred + obs
    values = np.full(pred.shape, np.nan, dtype=float)
    mask = denom > 0
    values[mask] = np.sqrt(2 * (pred[mask] - obs[mask]) ** 2 / denom[mask])
    return values


def geh_summary(predicted, observed) -> dict[str, float]:
    values = geh_values(predicted, observed)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"geh_median": math.nan, "geh_share_lt_5": math.nan, "geh_share_lt_10": math.nan}
    return {
        "geh_median": float(np.median(values)),
        "geh_share_lt_5": float(np.mean(values < 5)),
        "geh_share_lt_10": float(np.mean(values < 10)),
    }


def topk_overlap(predicted, observed, shares=(0.05, 0.10, 0.20)) -> dict[str, float]:
    pred, obs = _arrays(predicted, observed)
    n = pred.size
    if n == 0:
        return {f"top_{int(share * 100)}pct_overlap": math.nan for share in shares}
    out = {}
    pred_rank = np.argsort(pred)[::-1]
    obs_rank = np.argsort(obs)[::-1]
    for share in shares:
        k = max(1, int(math.ceil(n * share)))
        overlap = len(set(pred_rank[:k]).intersection(set(obs_rank[:k]))) / k
        out[f"top_{int(share * 100)}pct_overlap"] = float(overlap)
    return out


def calibration(predicted, observed) -> dict[str, float]:
    pred, obs = _arrays(predicted, observed)
    if pred.size < 2:
        return {"calibration_slope": math.nan, "calibration_intercept": math.nan}
    model = LinearRegression().fit(pred.reshape(-1, 1), obs)
    return {
        "calibration_slope": float(model.coef_[0]),
        "calibration_intercept": float(model.intercept_),
    }


def volume_share_jsd(predicted, observed) -> float:
    pred, obs = _arrays(predicted, observed)
    pred = np.maximum(pred, 0)
    obs = np.maximum(obs, 0)
    if pred.sum() <= 0 or obs.sum() <= 0:
        return math.nan
    return float(jensenshannon(pred / pred.sum(), obs / obs.sum()) ** 2)


def primary_metrics(predicted, observed) -> dict[str, Any]:
    pred, obs = _arrays(predicted, observed)
    if pred.size == 0:
        return {}
    mae = float(np.mean(np.abs(pred - obs)))
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    nrmse = rmse / float(np.mean(obs)) if np.mean(obs) else math.nan
    pearson = pearsonr(pred, obs).statistic if pred.size > 1 else math.nan
    spearman = spearmanr(pred, obs).statistic if pred.size > 1 else math.nan
    out: dict[str, Any] = {
        "weighted_mape": weighted_mape(pred, obs),
        "rmsle": rmsle(pred, obs),
        "mae": mae,
        "rmse": rmse,
        "normalized_rmse": float(nrmse),
        "pearson_correlation": float(pearson),
        "spearman_correlation": float(spearman),
        "bias_ratio": float(pred.sum() / obs.sum()) if obs.sum() else math.nan,
        "jensen_shannon_divergence": volume_share_jsd(pred, obs),
    }
    out.update(geh_summary(pred, obs))
    out.update(topk_overlap(pred, obs))
    out.update(calibration(pred, obs))
    return out


def metrics_by_group(
    frame: pd.DataFrame,
    *,
    predicted_col: str,
    observed_col: str,
    group_col: str,
) -> pd.DataFrame:
    rows = []
    for group, sub in frame.groupby(group_col, dropna=False):
        metrics = primary_metrics(sub[predicted_col], sub[observed_col])
        metrics[group_col] = group
        rows.append(metrics)
    return pd.DataFrame(rows)
