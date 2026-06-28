"""Guards for optional temporal-count validation."""

from __future__ import annotations

from typing import Any


class TemporalValidationNotSupported(RuntimeError):
    pass


def temporal_validation_supported(
    survey_audit: dict[str, Any], observed_manifest: list[dict[str, Any]]
) -> bool:
    if not survey_audit.get("usable_for_calendar_temporal_validation"):
        return False
    for manifest in observed_manifest:
        if manifest.get("temporal_type") in {"daily", "monthly", "hourly"}:
            temporal_range = manifest.get("temporal_range_found") or {}
            if temporal_range.get("date_min") or temporal_range.get("months"):
                return True
    return False


def require_temporal_overlap(
    survey_audit: dict[str, Any],
    observed_manifest: list[dict[str, Any]],
    *,
    mode: str,
) -> None:
    if mode != "temporal_overlap_counts":
        return
    if not temporal_validation_supported(survey_audit, observed_manifest):
        raise TemporalValidationNotSupported(
            "temporal_overlap_counts requires audited survey dates/months and an observed "
            "count source with overlapping daily/monthly/hourly coverage. Use "
            "spatial_aadt_proxy or relative_share_validation for HPMS AADT."
        )
