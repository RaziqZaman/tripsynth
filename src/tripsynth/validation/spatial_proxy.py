"""Spatial AADT proxy validation for routed baseline volumes."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from tripsynth.validation.metrics import primary_metrics


def allocate_desire_line_volume_to_observed(
    routed: gpd.GeoDataFrame,
    observed: gpd.GeoDataFrame,
    config: dict[str, Any],
) -> gpd.GeoDataFrame:
    projected_crs = config.get("project", {}).get("crs_projected", "EPSG:26918")
    buffer_m = float(config.get("validation", {}).get("baseline", {}).get("desire_line_buffer_m", 5000))

    observed_proj = observed.to_crs(projected_crs).reset_index(drop=True).copy()
    routed_proj = routed.to_crs(projected_crs).reset_index(drop=True).copy()
    observed_proj["observed_index"] = observed_proj.index

    if routed_proj.empty or observed_proj.empty:
        out = observed.copy()
        out["predicted_volume"] = 0.0
        out["allocated_match_count"] = 0
        out["match_method"] = "none"
        return out

    route_buffers = routed_proj[["route_id", "predicted_volume", "geometry"]].copy()
    route_buffers["geometry"] = routed_proj.geometry.buffer(buffer_m)
    pairs = gpd.sjoin(
        observed_proj[["observed_index", "geometry"]],
        route_buffers,
        how="inner",
        predicate="intersects",
    )

    contributions = []
    if not pairs.empty:
        pairs["candidate_weight"] = pairs.geometry.length
        for route_id, group in pairs.groupby("route_id"):
            route_volume = float(group["predicted_volume"].iloc[0])
            weights = group["candidate_weight"].to_numpy(dtype=float)
            if not np.isfinite(weights).all() or weights.sum() <= 0:
                weights = np.ones(len(group), dtype=float)
            allocated = route_volume * weights / weights.sum()
            for observed_index, value in zip(group["observed_index"], allocated):
                contributions.append(
                    {
                        "observed_index": int(observed_index),
                        "predicted_volume": float(value),
                        "match_method": "buffer_intersection",
                    }
                )
    else:
        nearest = gpd.sjoin_nearest(
            routed_proj[["route_id", "predicted_volume", "geometry"]],
            observed_proj[["observed_index", "geometry"]],
            how="left",
            distance_col="match_distance_m",
        )
        for row in nearest.dropna(subset=["observed_index"]).itertuples(index=False):
            contributions.append(
                {
                    "observed_index": int(row.observed_index),
                    "predicted_volume": float(row.predicted_volume),
                    "match_method": "nearest_fallback",
                }
            )

    out = observed.copy().reset_index(drop=True)
    if contributions:
        contrib = pd.DataFrame(contributions)
        by_segment = (
            contrib.groupby("observed_index")
            .agg(
                predicted_volume=("predicted_volume", "sum"),
                allocated_match_count=("predicted_volume", "size"),
                match_method=("match_method", lambda values: ",".join(sorted(set(values)))),
            )
            .reset_index()
        )
        out = out.merge(by_segment, left_index=True, right_on="observed_index", how="left")
        out = out.drop(columns=["observed_index"])
    else:
        out["predicted_volume"] = 0.0
        out["allocated_match_count"] = 0
        out["match_method"] = "none"

    out["predicted_volume"] = pd.to_numeric(out["predicted_volume"], errors="coerce").fillna(0.0)
    out["allocated_match_count"] = (
        pd.to_numeric(out["allocated_match_count"], errors="coerce").fillna(0).astype(int)
    )
    out["match_method"] = out["match_method"].fillna("unmatched")
    return gpd.GeoDataFrame(out, geometry="geometry", crs=observed.crs)


def spatial_aadt_proxy_metrics(segment_frame: gpd.GeoDataFrame) -> pd.DataFrame:
    frame = segment_frame.dropna(subset=["observed_aadt"]).copy()
    frame["observed_aadt"] = pd.to_numeric(frame["observed_aadt"], errors="coerce")
    frame["predicted_volume"] = pd.to_numeric(frame["predicted_volume"], errors="coerce").fillna(0)
    frame = frame.dropna(subset=["observed_aadt"])
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame(rows)

    pred = frame["predicted_volume"].to_numpy(dtype=float)
    obs = frame["observed_aadt"].to_numpy(dtype=float)
    rows.append({"comparison_type": "uncalibrated_absolute_proxy", **primary_metrics(pred, obs)})

    if pred.sum() > 0 and obs.sum() > 0:
        calibrated = pred * (obs.sum() / pred.sum())
        rows.append({"comparison_type": "sum_calibrated_absolute_proxy", **primary_metrics(calibrated, obs)})
        pred_share = pred / pred.sum()
        obs_share = obs / obs.sum()
        share_metrics = primary_metrics(pred_share, obs_share)
        rows.append(
            {
                "comparison_type": "relative_share_validation",
                **{f"share_{key}": value for key, value in share_metrics.items()},
            }
        )
    return pd.DataFrame(rows)


def _direct_segment_match(
    routed: gpd.GeoDataFrame, observed: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    route_sum = (
        routed.assign(observed_segment_id=routed["observed_segment_id"].astype("string"))
        .groupby("observed_segment_id", dropna=False)["predicted_volume"]
        .sum()
        .reset_index()
    )
    out = observed.copy().reset_index(drop=True)
    out["observed_segment_id"] = out["segment_id"].astype("string")
    out = out.merge(route_sum, on="observed_segment_id", how="left")
    out["predicted_volume"] = pd.to_numeric(out["predicted_volume"], errors="coerce").fillna(0.0)
    out["allocated_match_count"] = (out["predicted_volume"] > 0).astype(int)
    out["match_method"] = out["allocated_match_count"].map({1: "direct_network_segment", 0: "unmatched"})
    return gpd.GeoDataFrame(out, geometry="geometry", crs=observed.crs)


def validate_spatial_aadt_proxy(
    routed: gpd.GeoDataFrame,
    observed: gpd.GeoDataFrame,
    config: dict[str, Any],
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    by_segment = (
        _direct_segment_match(routed, observed)
        if "observed_segment_id" in routed.columns
        else allocate_desire_line_volume_to_observed(routed, observed, config)
    )
    metrics = spatial_aadt_proxy_metrics(by_segment)
    return by_segment, metrics
