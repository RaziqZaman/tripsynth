from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trip_synth.data.preprocessing import FittedPreprocessor
from trip_synth.data.schema import FeatureSchema
from trip_synth.utils.io import ensure_dir, write_json
from trip_synth.utils.progress import progress_iter

from .metrics import jensen_shannon, total_variation


DEFAULT_CROSSES = [
    ("o_activity", "d_activity"),
    ("travel_mode", "departure_time_bin"),
    ("o_county_fips", "d_county_fips"),
    ("tdate_dow", "departure_time_bin"),
    ("hh_income_detailed", "vehicle_occupancy"),
    ("o_activity", "travel_mode"),
    ("d_activity", "travel_mode"),
    ("distance_bin", "reported_travel_time_bin"),
    ("year", "vehicle"),
]


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "departure_time_minutes" in out.columns:
        dep = pd.to_numeric(out["departure_time_minutes"], errors="coerce")
        out["departure_time_bin"] = pd.cut(
            dep,
            bins=[-1, 360, 600, 900, 1200, 1441],
            labels=["night", "am_peak", "midday", "pm_peak", "evening"],
        ).astype("string").fillna("missing")
    if "reported_travel_time" in out.columns:
        rt = pd.to_numeric(out["reported_travel_time"], errors="coerce")
        out["reported_travel_time_bin"] = pd.cut(
            rt,
            bins=[-1, 10, 20, 40, 60, 120, 10000],
            labels=["0_10", "10_20", "20_40", "40_60", "60_120", "120_plus"],
        ).astype("string").fillna("missing")
    if "distance" in out.columns:
        dist = pd.to_numeric(out["distance"], errors="coerce")
        out["distance_bin"] = pd.cut(
            dist,
            bins=[-1, 1, 3, 7, 15, 30, 10000],
            labels=["0_1", "1_3", "3_7", "7_15", "15_30", "30_plus"],
        ).astype("string").fillna("missing")
    for stem in ["o", "d", "home", "work"]:
        col = f"{stem}_tract_fips"
        if col in out.columns:
            out[f"{stem}_county_fips"] = out[col].astype(str).str[:5]
    return out


def _weighted_cross(df: pd.DataFrame, a: str, b: str, weights: pd.Series | None) -> pd.Series:
    vals = pd.DataFrame({"a": df[a].astype("string"), "b": df[b].astype("string")})
    if weights is None:
        counts = vals.groupby(["a", "b"], dropna=False).size().astype(float)
    else:
        vals["weight"] = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0).to_numpy(float)
        counts = vals.groupby(["a", "b"], dropna=False)["weight"].sum()
    total = float(counts.sum())
    return counts / total if total > 0 else counts


def _plot_cross_error(real: pd.Series, synth: pd.Series, title: str, out: Path) -> None:
    keys = sorted(set(real.index) | set(synth.index), key=lambda k: str(k))[:60]
    if not keys:
        return
    rows = sorted({k[0] for k in keys})
    cols = sorted({k[1] for k in keys})
    mat = np.zeros((len(rows), len(cols)))
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            mat[i, j] = synth.get((r, c), 0.0) - real.get((r, c), 0.0)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(mat, cmap="coolwarm", aspect="auto")
    plt.colorbar(im, label="Synthetic - survey share")
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=7)
    plt.yticks(range(len(rows)), rows, fontsize=7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def validate_method_cross_marginals(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    schema: FeatureSchema,
    preprocessor: FittedPreprocessor,
    method: str,
    run_dir: str | Path,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    fig_dir = ensure_dir(run_dir / "figures" / "cross_marginals" / method)
    real = add_derived_columns(real_df)
    synth = add_derived_columns(synthetic_df)
    weights = real[schema.weight_column] if schema.weight_column in real.columns else None
    metrics: dict[str, Any] = {"method": method, "crosses": {}}
    crosses = list(DEFAULT_CROSSES)
    for a, b in progress_iter(crosses, desc=f"{method} cross-marginals", total=len(crosses), unit="cross"):
        if a not in real.columns or b not in real.columns or a not in synth.columns or b not in synth.columns:
            continue
        rp = _weighted_cross(real, a, b, weights)
        sp = _weighted_cross(synth, a, b, None)
        keys = sorted(set(rp.index) | set(sp.index), key=lambda k: str(k))
        p = np.array([rp.get(k, 0.0) for k in keys], dtype=float)
        q = np.array([sp.get(k, 0.0) for k in keys], dtype=float)
        errors = sorted(
            [{"cell": f"{k[0]}|{k[1]}", "abs_error": float(abs(p[i] - q[i]))} for i, k in enumerate(keys)],
            key=lambda row: row["abs_error"],
            reverse=True,
        )[:10]
        name = f"{a}__x__{b}"
        metrics["crosses"][name] = {
            "total_variation": total_variation(p, q),
            "jensen_shannon": jensen_shannon(p, q),
            "top_cell_abs_errors": errors,
        }
        _plot_cross_error(rp, sp, f"{method}: {a} x {b}", fig_dir / f"{name}.png")
    tvs = [row["total_variation"] for row in metrics["crosses"].values()]
    metrics["summary"] = {
        "mean_cross_tv": float(np.mean(tvs)) if tvs else float("nan"),
        "cross_marginal_similarity_score": float(1.0 - np.nanmean(tvs)) if tvs else 0.0,
    }
    write_json(metrics, run_dir / "metrics" / f"{method}_cross_marginals.json")
    return metrics
