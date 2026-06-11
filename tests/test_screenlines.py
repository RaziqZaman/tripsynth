from __future__ import annotations

import numpy as np

from trip_synth.validation.aadt_validation import compute_aadt_metrics
from trip_synth.validation.route_assignment import adjacent_pairs_from_ordered_tracts, screenline_id


def test_screenline_ids_are_undirected() -> None:
    assert screenline_id("b", "a") == screenline_id("a", "b")
    assert adjacent_pairs_from_ordered_tracts(["1", "2", "2", "3"]) == [("1", "2"), ("2", "3")]


def test_aadt_metrics_include_geh() -> None:
    metrics = compute_aadt_metrics(np.array([1000, 2000, 3000]), np.array([980, 2200, 2800]))
    assert metrics["screenlines"] == 3
    assert "share_geh_lt_5" in metrics
    assert 0 <= metrics["share_geh_lt_5"] <= 1


def test_virtual_counts_average_weekday() -> None:
    import pandas as pd

    from trip_synth.validation.aadt_validation import _synthetic_average_screenline_counts

    synthetic = pd.DataFrame(
        {
            "o_tract_fips": ["24001000100", "24001000100", "24001000100", "24001000100"],
            "d_tract_fips": ["24001000200", "24001000200", "24001000200", "24001000200"],
            "tdate_dow": [1, 1, 2, 6],
        }
    )
    paths = pd.DataFrame(
        {
            "o_tract_fips": ["24001000100"],
            "d_tract_fips": ["24001000200"],
            "path_position": [0],
            "screenline_id": ["24001000100__24001000200"],
        }
    )
    counts = _synthetic_average_screenline_counts(synthetic, paths, "average_weekday")
    assert counts.loc[0, "screenline_id"] == "24001000100__24001000200"
    assert counts.loc[0, "synthetic_count"] == 1.5
