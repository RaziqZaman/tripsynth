"""Survey-internal holdout validation that does not require observed counts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from tripsynth.routing.desire_lines import tract_to_county_fips


DEFAULT_FEATURES = [
    "origin_county_fips",
    "destination_county_fips",
    "od_county_pair",
    "mode",
    "purpose",
    "departure_hour",
    "day_of_week",
    "survey_month",
]


@dataclass(frozen=True)
class SurveyHoldoutSplit:
    train: pd.DataFrame
    holdout: pd.DataFrame
    metadata: dict[str, Any]


def _nonempty_field(frame: pd.DataFrame, field: str) -> bool:
    return field in frame and frame[field].notna().any()


def split_survey_holdout(
    trips: pd.DataFrame,
    config: dict[str, Any],
    *,
    seed: int,
) -> SurveyHoldoutSplit:
    holdout_config = config.get("validation", {}).get("survey_holdout", {})
    holdout_share = float(holdout_config.get("holdout_share", 0.2))
    group_preference = holdout_config.get(
        "split_group_preference", ["household_id", "person_id"]
    )
    rng = np.random.default_rng(seed)

    group_field = next((field for field in group_preference if _nonempty_field(trips, field)), None)
    if group_field:
        groups = pd.Series(trips[group_field].dropna().unique())
        holdout_n = max(1, int(round(len(groups) * holdout_share)))
        holdout_groups = set(rng.choice(groups.to_numpy(), size=holdout_n, replace=False))
        mask = trips[group_field].isin(holdout_groups)
        split_method = f"grouped_by_{group_field}"
    else:
        mask = pd.Series(False, index=trips.index)
        holdout_n = max(1, int(round(len(trips) * holdout_share)))
        holdout_idx = rng.choice(trips.index.to_numpy(), size=holdout_n, replace=False)
        mask.loc[holdout_idx] = True
        split_method = "row_random"

    train = trips.loc[~mask].copy()
    holdout = trips.loc[mask].copy()
    return SurveyHoldoutSplit(
        train=train,
        holdout=holdout,
        metadata={
            "split_method": split_method,
            "holdout_share": holdout_share,
            "train_rows": int(len(train)),
            "holdout_rows": int(len(holdout)),
            "seed": seed,
        },
    )


def add_holdout_validation_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["origin_county_fips"] = out["origin_tract"].map(tract_to_county_fips)
    out["destination_county_fips"] = out["destination_tract"].map(tract_to_county_fips)
    out["od_county_pair"] = (
        out["origin_county_fips"].astype("string")
        + "_"
        + out["destination_county_fips"].astype("string")
    )

    departure = pd.to_numeric(out.get("departure_time"), errors="coerce")
    out["departure_hour"] = (departure.where(departure.between(0, 1439)) // 60).astype("Int64")

    dates = pd.to_datetime(out.get("trip_date"), errors="coerce")
    out["survey_month"] = dates.dt.month.astype("Int64")
    return out


def _weighted_distribution(frame: pd.DataFrame, feature: str, weight_field: str) -> pd.Series:
    if feature not in frame:
        return pd.Series(dtype=float)
    working = frame[[feature]].copy()
    working[feature] = working[feature].astype("string").fillna("__missing__")
    if weight_field in frame:
        weights = pd.to_numeric(frame[weight_field], errors="coerce").fillna(0)
        weights = weights.where(weights > 0, 0)
    else:
        weights = pd.Series(1.0, index=frame.index)
    grouped = weights.groupby(working[feature]).sum()
    total = grouped.sum()
    if total <= 0:
        grouped = working[feature].value_counts().astype(float)
        total = grouped.sum()
    return grouped / total if total > 0 else grouped


def _feature_metrics(
    synthetic: pd.DataFrame,
    holdout: pd.DataFrame,
    feature: str,
    *,
    weight_field: str,
    top_k: int,
) -> dict[str, Any]:
    syn = _weighted_distribution(synthetic, feature, weight_field)
    obs = _weighted_distribution(holdout, feature, weight_field)
    categories = syn.index.union(obs.index)
    syn_values = syn.reindex(categories, fill_value=0).to_numpy(dtype=float)
    obs_values = obs.reindex(categories, fill_value=0).to_numpy(dtype=float)
    total_variation = 0.5 * np.abs(syn_values - obs_values).sum()
    jsd = jensenshannon(syn_values, obs_values) ** 2 if syn_values.sum() and obs_values.sum() else np.nan
    top_syn = set(syn.sort_values(ascending=False).head(top_k).index)
    top_obs = set(obs.sort_values(ascending=False).head(top_k).index)
    top_overlap = len(top_syn & top_obs) / max(1, min(top_k, len(top_obs)))
    support_overlap = len(set(syn.index) & set(obs.index)) / max(1, len(set(obs.index)))
    return {
        "feature": feature,
        "synthetic_categories": int(len(syn)),
        "holdout_categories": int(len(obs)),
        "jensen_shannon_divergence": float(jsd),
        "total_variation_distance": float(total_variation),
        "top_k_overlap": float(top_overlap),
        "holdout_support_covered": float(support_overlap),
    }


def validate_synthetic_against_survey_holdout(
    synthetic: pd.DataFrame,
    holdout: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout_config = config.get("validation", {}).get("survey_holdout", {})
    features = holdout_config.get("features", DEFAULT_FEATURES)
    top_k = int(holdout_config.get("top_k", 10))
    weight_field = holdout_config.get("weight_field", "weight")

    syn = add_holdout_validation_features(synthetic)
    obs = add_holdout_validation_features(holdout)
    rows = [
        _feature_metrics(syn, obs, feature, weight_field=weight_field, top_k=top_k)
        for feature in features
    ]
    by_feature = pd.DataFrame(rows)
    by_run = pd.DataFrame(
        [
            {
                "validation_mode": "survey_holdout",
                "features_evaluated": int(len(by_feature)),
                "mean_jensen_shannon_divergence": float(
                    by_feature["jensen_shannon_divergence"].mean()
                ),
                "mean_total_variation_distance": float(
                    by_feature["total_variation_distance"].mean()
                ),
                "mean_top_k_overlap": float(by_feature["top_k_overlap"].mean()),
                "mean_holdout_support_covered": float(
                    by_feature["holdout_support_covered"].mean()
                ),
            }
        ]
    )
    return by_feature, by_run
