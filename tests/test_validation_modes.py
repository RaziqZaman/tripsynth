import pytest

from tripsynth.validation.temporal_alignment import (
    TemporalValidationNotSupported,
    require_temporal_overlap,
)


def test_temporal_validation_refuses_without_overlap():
    audit = {"usable_for_calendar_temporal_validation": False}
    manifest = [{"temporal_type": "annual_average", "temporal_range_found": {"years": [2018]}}]
    with pytest.raises(TemporalValidationNotSupported):
        require_temporal_overlap(audit, manifest, mode="temporal_overlap_counts")


def test_spatial_proxy_mode_does_not_require_overlap():
    require_temporal_overlap({}, [], mode="spatial_aadt_proxy")
