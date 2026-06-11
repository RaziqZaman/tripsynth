from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trip_synth.data.load import read_survey_csv
from trip_synth.data.schema import FeatureSchema, clean_fips_value
from trip_synth.utils.io import ensure_dir, load_yaml, write_json

from .aadt_screenlines import build_screenlines, load_tracts
from .metrics import geh, regression_slope_intercept, safe_corr
from .route_assignment import screenline_id


def compute_aadt_metrics(observed: np.ndarray, synthetic: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    synthetic = np.asarray(synthetic, dtype=float)
    mask = np.isfinite(observed) & np.isfinite(synthetic)
    if mask.sum() == 0:
        return {"screenlines": 0}
    o = observed[mask]
    s = synthetic[mask]
    g = geh(o, s)
    log_o = np.log1p(o)
    log_s = np.log1p(s)
    slope, intercept = regression_slope_intercept(log_o, log_s)
    denom = np.maximum(np.abs(o), 1.0)
    ss_res = float(np.sum((s - o) ** 2))
    ss_tot = float(np.sum((o - np.mean(o)) ** 2))
    log_ss_res = float(np.sum((log_s - log_o) ** 2))
    log_ss_tot = float(np.sum((log_o - np.mean(log_o)) ** 2))
    return {
        "screenlines": int(mask.sum()),
        "pearson_raw": safe_corr(o, s, "pearson"),
        "spearman_raw": safe_corr(o, s, "spearman"),
        "pearson_log1p": safe_corr(log_o, log_s, "pearson"),
        "spearman_log1p": safe_corr(log_o, log_s, "spearman"),
        "rmse": float(np.sqrt(np.mean((s - o) ** 2))),
        "mae": float(np.mean(np.abs(s - o))),
        "mape_guarded": float(np.mean(np.abs(s - o) / denom)),
        "smape": float(np.mean(2 * np.abs(s - o) / np.maximum(np.abs(s) + np.abs(o), 1.0))),
        "median_ape": float(np.median(np.abs(s - o) / denom)),
        "bias_mean_error": float(np.mean(s - o)),
        "r2_raw": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "r2_log1p": float(1.0 - log_ss_res / log_ss_tot) if log_ss_tot > 0 else float("nan"),
        "geh_mean": float(np.mean(g)),
        "share_geh_lt_5": float(np.mean(g < 5)),
        "share_geh_lt_10": float(np.mean(g < 10)),
        "calibration_slope_log1p": slope,
        "calibration_intercept_log1p": intercept,
    }


def _clean_tract_series(series: pd.Series) -> pd.Series:
    return series.map(clean_fips_value).astype(str)


def _candidate_od_pairs(config: dict[str, Any], run_dir: Path) -> pd.DataFrame:
    schema = FeatureSchema.from_yaml("configs/schema.yaml")
    real_df, _ = read_survey_csv(config["input_csv"], schema)
    frames = []
    if {"o_tract_fips", "d_tract_fips"}.issubset(real_df.columns):
        frames.append(real_df[["o_tract_fips", "d_tract_fips"]])
    for method in config.get("methods", []):
        sample_path = run_dir / "samples" / f"{method}_synthetic.csv"
        if sample_path.exists():
            df = pd.read_csv(sample_path, usecols=lambda c: c in {"o_tract_fips", "d_tract_fips"}, low_memory=False)
            if {"o_tract_fips", "d_tract_fips"}.issubset(df.columns):
                frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["o_tract_fips", "d_tract_fips"])
    od = pd.concat(frames, ignore_index=True)
    od["o_tract_fips"] = _clean_tract_series(od["o_tract_fips"])
    od["d_tract_fips"] = _clean_tract_series(od["d_tract_fips"])
    od = od[(od["o_tract_fips"] != "-1") & (od["d_tract_fips"] != "-1")]
    od = od.drop_duplicates()
    max_pairs = int(config.get("aadt_validation", {}).get("max_od_pairs", 250000))
    if len(od) > max_pairs:
        od = od.head(max_pairs)
    return od.reset_index(drop=True)


def _ordered_crossed_tracts(line, candidates, origin: str, dest: str) -> list[str]:
    rows = []
    for tract, geom in candidates[["GEOID", "geometry"]].itertuples(index=False):
        try:
            inter = geom.intersection(line)
        except Exception:
            continue
        if inter.is_empty:
            continue
        point = inter.representative_point()
        pos = float(line.project(point))
        rows.append((pos, str(tract)))
    rows.extend([(0.0, origin), (float(line.length), dest)])
    rows = sorted(rows, key=lambda x: x[0])
    ordered: list[str] = []
    for _, tract in rows:
        if not ordered or ordered[-1] != tract:
            ordered.append(tract)
    if ordered and ordered[0] != origin:
        ordered.insert(0, origin)
    if ordered and ordered[-1] != dest:
        ordered.append(dest)
    return ordered


def build_od_screenline_paths(config: dict[str, Any], run_dir: str | Path):
    import geopandas as gpd
    from shapely.geometry import LineString

    run_dir = Path(run_dir)
    out_path = run_dir / "geo" / "od_screenline_paths.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)
    screenline_path = run_dir / "geo" / "screenlines.parquet"
    if not screenline_path.exists():
        status = build_screenlines(config, run_dir)
        if status.get("status") != "ok":
            raise FileNotFoundError("screenlines.parquet is unavailable and could not be built")
    tracts = load_tracts(config, run_dir)
    screenlines = gpd.read_parquet(screenline_path)
    valid_screenlines = set(screenlines["screenline_id"].astype(str))
    od = _candidate_od_pairs(config, run_dir)
    if od.empty:
        empty = pd.DataFrame(columns=["o_tract_fips", "d_tract_fips", "path_position", "screenline_id"])
        empty.to_parquet(out_path)
        return empty
    tract_lookup = tracts.set_index("GEOID")
    sindex = tracts.sindex
    path_rows: list[dict[str, Any]] = []
    for origin, dest in od[["o_tract_fips", "d_tract_fips"]].itertuples(index=False):
        if origin == dest or origin not in tract_lookup.index or dest not in tract_lookup.index:
            continue
        a = tract_lookup.loc[origin]
        b = tract_lookup.loc[dest]
        line = LineString([(float(a.rep_x), float(a.rep_y)), (float(b.rep_x), float(b.rep_y))])
        candidate_idx = list(sindex.query(line, predicate="intersects"))
        candidates = tracts.iloc[candidate_idx]
        ordered = _ordered_crossed_tracts(line, candidates, origin, dest)
        position = 0
        for left, right in zip(ordered[:-1], ordered[1:]):
            sid = screenline_id(left, right)
            if sid not in valid_screenlines:
                continue
            path_rows.append(
                {
                    "o_tract_fips": origin,
                    "d_tract_fips": dest,
                    "path_position": position,
                    "screenline_id": sid,
                }
            )
            position += 1
    paths = pd.DataFrame(path_rows)
    if paths.empty:
        paths = pd.DataFrame(columns=["o_tract_fips", "d_tract_fips", "path_position", "screenline_id"])
    paths.to_parquet(out_path)
    return paths


def _weekday_values(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric.notna(), series.astype(str))


def _synthetic_average_screenline_counts(
    synthetic: pd.DataFrame,
    paths: pd.DataFrame,
    comparison_basis: str,
) -> pd.DataFrame:
    if paths.empty or synthetic.empty:
        return pd.DataFrame(columns=["screenline_id", "synthetic_count"])
    trips = synthetic[["o_tract_fips", "d_tract_fips"]].copy()
    trips["trip_id"] = np.arange(len(trips))
    trips["o_tract_fips"] = _clean_tract_series(trips["o_tract_fips"])
    trips["d_tract_fips"] = _clean_tract_series(trips["d_tract_fips"])
    if "tdate_dow" in synthetic.columns:
        trips["tdate_dow"] = _weekday_values(synthetic["tdate_dow"])
    expanded = trips.merge(paths, on=["o_tract_fips", "d_tract_fips"], how="inner")
    if expanded.empty:
        return pd.DataFrame(columns=["screenline_id", "synthetic_count"])
    if comparison_basis == "average_weekday" and "tdate_dow" in expanded.columns:
        dow = pd.to_numeric(expanded["tdate_dow"], errors="coerce")
        expanded = expanded[dow.between(1, 5)]
        if expanded.empty:
            return pd.DataFrame(columns=["screenline_id", "synthetic_count"])
        by_dow = expanded.groupby(["screenline_id", "tdate_dow"]).size().reset_index(name="daily_count")
        counts = by_dow.groupby("screenline_id")["daily_count"].mean().reset_index(name="synthetic_count")
    elif "tdate_dow" in expanded.columns:
        by_dow = expanded.groupby(["screenline_id", "tdate_dow"]).size().reset_index(name="daily_count")
        counts = by_dow.groupby("screenline_id")["daily_count"].mean().reset_index(name="synthetic_count")
    else:
        counts = expanded.groupby("screenline_id").size().reset_index(name="synthetic_count")
    return counts


def build_virtual_screenline_counts(config: dict[str, Any], run_dir: str | Path) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    paths = build_od_screenline_paths(config, run_dir)
    screenlines = pd.read_parquet(run_dir / "geo" / "screenlines.parquet")
    observed_field = str(screenlines.get("observed_count_field", pd.Series([""])).dropna().astype(str).iloc[0]) if len(screenlines) else ""
    comparison_basis = "average_weekday" if observed_field == "AAWDT" else "average_day"
    outputs: dict[str, pd.DataFrame] = {}
    for method in config.get("methods", []):
        sample_path = run_dir / "samples" / f"{method}_synthetic.csv"
        if not sample_path.exists():
            continue
        synthetic = pd.read_csv(sample_path, low_memory=False)
        counts = _synthetic_average_screenline_counts(synthetic, paths, comparison_basis)
        counts["method"] = method
        counts["comparison_basis"] = comparison_basis
        out_path = run_dir / "metrics" / f"{method}_virtual_screenline_counts.parquet"
        counts.to_parquet(out_path)
        outputs[str(method)] = counts
    return outputs


def _plot_aadt_outputs(run_dir: Path, comparisons: pd.DataFrame, screenlines) -> None:
    ensure_dir(run_dir / "figures" / "poster")
    if comparisons.empty:
        return
    plt.figure(figsize=(7, 6))
    for method, group in comparisons.groupby("method"):
        plt.scatter(np.log1p(group["observed_count"]), np.log1p(group["synthetic_count"]), s=18, alpha=0.55, label=method)
    plt.xlabel("log1p observed AADT/AAWDT")
    plt.ylabel("log1p synthetic virtual crossings")
    plt.title("Observed vs synthetic screenline volumes")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(run_dir / "figures" / "poster" / "aadt_observed_vs_synthetic_log_scatter.png", dpi=300)
    plt.savefig(run_dir / "figures" / "poster" / "08_aadt_observed_vs_synthetic_log_scatter.png", dpi=300)
    plt.close()

    leaderboard = comparisons.groupby("method").apply(
        lambda g: compute_aadt_metrics(g["observed_count"].to_numpy(), g["synthetic_count"].to_numpy()).get("pearson_log1p", np.nan),
        include_groups=False,
    ).reset_index(name="pearson_log1p")
    plt.figure(figsize=(8, 4.8))
    plt.bar(leaderboard["method"], leaderboard["pearson_log1p"].fillna(0.0))
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Pearson correlation, log1p")
    plt.title("AADT validation leaderboard")
    plt.tight_layout()
    plt.savefig(run_dir / "figures" / "poster" / "aadt_method_leaderboard.png", dpi=300)
    plt.savefig(run_dir / "figures" / "poster" / "09_aadt_method_leaderboard.png", dpi=300)
    plt.close()

    comparisons["geh"] = geh(comparisons["observed_count"].to_numpy(), comparisons["synthetic_count"].to_numpy())
    plt.figure(figsize=(8, 4.8))
    for method, group in comparisons.groupby("method"):
        plt.hist(group["geh"], bins=30, alpha=0.45, label=method)
    plt.xlabel("GEH")
    plt.ylabel("Screenlines")
    plt.title("GEH distribution by method")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(run_dir / "figures" / "poster" / "geh_distribution_by_method.png", dpi=300)
    plt.savefig(run_dir / "figures" / "poster" / "10_geh_distribution_by_method.png", dpi=300)
    plt.close()

    contrastive = comparisons[comparisons["method"] == "contrastive_vae"]
    if not contrastive.empty and hasattr(screenlines, "plot"):
        import geopandas as gpd

        mapped = screenlines.merge(
            contrastive[["screenline_id", "residual"]], on="screenline_id", how="left"
        )
        mapped = gpd.GeoDataFrame(mapped, geometry="geometry", crs=screenlines.crs)
        ax = mapped.dropna(subset=["residual"]).plot(
            column="residual", cmap="coolwarm", legend=True, figsize=(8, 8), linewidth=1.2
        )
        ax.set_axis_off()
        ax.set_title("Contrastive VAE screenline residuals")
        plt.tight_layout()
        plt.savefig(run_dir / "figures" / "poster" / "aadt_residual_map_contrastive_vae.png", dpi=300)
        plt.close()


def run_aadt_validation(config: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    import geopandas as gpd

    run_dir = Path(run_dir)
    ensure_dir(run_dir / "metrics")
    ensure_dir(run_dir / "tables")
    screenlines_path = run_dir / "geo" / "screenlines.parquet"
    if not screenlines_path.exists():
        status = build_screenlines(config, run_dir)
        if status.get("status") != "ok":
            result = {
                "status": "skipped",
                "reason": "screenlines.parquet is unavailable",
                "screenline_status": status,
                "temporal_framing": "Public AADT/AAWDT are average-day or average-weekday measures, not exact daily validation.",
            }
            write_json(result, run_dir / "metrics" / "aadt_validation_skipped.json")
            return result
    screenlines = gpd.read_parquet(screenlines_path)
    usable = screenlines[(screenlines["station_count"] > 0) & (screenlines["observed_count"] > 0)].copy()
    if usable.empty:
        result = {
            "status": "skipped",
            "reason": "No screenlines have mapped stations with positive observed counts",
            "temporal_framing": "Public AADT/AAWDT are average-day or average-weekday measures, not exact daily validation.",
        }
        write_json(result, run_dir / "metrics" / "aadt_validation_skipped.json")
        return result

    virtual = build_virtual_screenline_counts(config, run_dir)
    comparison_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for method, counts in virtual.items():
        merged = usable[["screenline_id", "tract_a", "tract_b", "observed_count", "observed_count_field", "station_count"]].merge(
            counts[["screenline_id", "synthetic_count", "comparison_basis"]], on="screenline_id", how="left"
        )
        merged["synthetic_count"] = merged["synthetic_count"].fillna(0.0)
        basis = str(counts["comparison_basis"].dropna().iloc[0]) if "comparison_basis" in counts and counts["comparison_basis"].notna().any() else ("average_weekday" if str(merged["observed_count_field"].dropna().iloc[0]) == "AAWDT" else "average_day")
        merged["comparison_basis"] = merged["comparison_basis"].fillna(basis)
        merged["method"] = method
        merged["residual"] = merged["synthetic_count"] - merged["observed_count"]
        merged["residual_pct_guarded"] = merged["residual"] / merged["observed_count"].clip(lower=1.0)
        comparison_rows.append(merged)
        metrics = compute_aadt_metrics(merged["observed_count"].to_numpy(), merged["synthetic_count"].to_numpy())
        metrics.update(
            {
                "method": method,
                "comparison_basis": str(merged["comparison_basis"].dropna().iloc[0]) if merged["comparison_basis"].notna().any() else "average_day",
                "observed_count_field": str(merged["observed_count_field"].dropna().iloc[0]),
            }
        )
        summary_rows.append(metrics)
    if not comparison_rows:
        result = {"status": "skipped", "reason": "No synthetic sample files were available for virtual counts"}
        write_json(result, run_dir / "metrics" / "aadt_validation_skipped.json")
        return result

    comparisons = pd.concat(comparison_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(run_dir / "tables" / "aadt_validation_summary.csv", index=False)
    residuals = comparisons.assign(abs_residual=lambda d: d["residual"].abs()).sort_values("abs_residual", ascending=False)
    residuals.head(20).to_csv(run_dir / "tables" / "aadt_top_residual_screenlines.csv", index=False)
    comparisons.to_parquet(run_dir / "metrics" / "aadt_screenline_comparisons.parquet")
    _plot_aadt_outputs(run_dir, comparisons, usable)
    result = {
        "status": "ok",
        "screenlines_compared": int(usable["screenline_id"].nunique()),
        "methods": sorted(virtual.keys()),
        "temporal_framing": "AADT/AAWDT are average traffic-volume measures. AAWDT is compared to average weekday synthetic virtual crossings when available; otherwise AADT/average-day framing is used.",
        "assignment_note": "OD-to-screenline paths use centroid-line tract crossings as a geometric proxy, not true route assignment.",
    }
    write_json(result, run_dir / "metrics" / "aadt_validation_summary.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medium.yaml")
    parser.add_argument("--run-dir", default="outputs/runs/manual_aadt")
    args = parser.parse_args()
    config = load_yaml(args.config)
    print(run_aadt_validation(config, args.run_dir))
