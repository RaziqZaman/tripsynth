import numpy as np

from tripsynth.validation.metrics import geh_summary, primary_metrics, rmsle, weighted_mape


def test_metric_equations_for_equal_arrays():
    predicted = np.array([100, 200, 300])
    observed = np.array([100, 200, 300])
    assert weighted_mape(predicted, observed) == 0
    assert rmsle(predicted, observed) == 0
    metrics = primary_metrics(predicted, observed)
    assert metrics["rmse"] == 0
    assert metrics["spearman_correlation"] == 1
    assert metrics["top_5pct_overlap"] == 1


def test_geh_summary():
    summary = geh_summary([100, 200], [100, 100])
    assert summary["geh_median"] > 0
    assert 0 <= summary["geh_share_lt_10"] <= 1
