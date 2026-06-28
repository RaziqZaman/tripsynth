"""Observed-count loading policy for experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd

from tripsynth.data_sources.observed_counts.fhwa_hpms import fetch_fhwa_hpms_2018
from tripsynth.data_sources.observed_counts.local_counts import load_local_observed_counts
from tripsynth.data_sources.observed_counts.mdot_sha import fetch_mdot_sha_aadt


def load_observed_counts_for_validation(
    config: dict[str, Any],
    *,
    source_name: str | None = None,
    force_fetch: bool = False,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Load normalized observed counts from local config, cache, or HPMS fetch."""
    observed_config = config.get("observed_counts", {})
    sources = observed_config.get("sources", {})
    requested = source_name or observed_config.get("default_source", "mdot_sha_aadt")

    if requested in sources and sources[requested].get("type") == "local_file":
        gdf = load_local_observed_counts(config, source_name=requested)
        return gdf, {"source": requested, "load_mode": "local_file"}

    if requested == "mdot_sha_aadt":
        processed = Path("data/processed/observed_counts/mdot_sha_aadt/observed_counts_mdot_sha_study_area.geoparquet")
        if processed.exists() and not force_fetch:
            return gpd.read_parquet(processed), {
                "source": "mdot_sha_aadt",
                "load_mode": "cached_processed",
                "path": str(processed),
            }
        result = fetch_mdot_sha_aadt(config, force=force_fetch)
        return result.filtered, {
            "source": "mdot_sha_aadt",
            "load_mode": "fetched" if not result.used_cache else "cached_fetch_result",
            "paths": result.output_paths,
        }

    local_user = sources.get("local_user_counts", {})
    if local_user.get("enabled") and (local_user.get("path") or local_user.get("local_path")):
        gdf = load_local_observed_counts(config, source_name="local_user_counts")
        return gdf, {"source": "local_user_counts", "load_mode": "local_file"}

    processed = Path(
        "data/processed/observed_counts/fhwa_hpms_2018/"
        "observed_counts_hpms_2018_dmv.geoparquet"
    )
    if processed.exists() and not force_fetch:
        return gpd.read_parquet(processed), {
            "source": "fhwa_hpms_2018_dmv",
            "load_mode": "cached_processed",
            "path": str(processed),
        }

    result = fetch_fhwa_hpms_2018(config, force=force_fetch)
    return result.filtered, {
        "source": "fhwa_hpms_2018_dmv",
        "load_mode": "fetched" if not result.used_cache else "cached_fetch_result",
        "paths": result.output_paths,
    }
