import pandas as pd

from tripsynth.synthesis.weighted_resampling import sample_by_scale


def test_weighted_resampling_smoke():
    trips = pd.DataFrame({"origin_tract": [1, 2], "destination_tract": [3, 4], "weight": [1, 10]})
    result = sample_by_scale(trips, scale_factor=2, seed=7)
    assert result.method == "weighted_resampling"
    assert len(result.synthetic_trips) == 4
    assert "synthetic_trip_id" in result.synthetic_trips
