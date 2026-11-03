import numpy as np
import pandas as pd

from anomaly_detection.inference.monitoring import monitoring_summary


def test_monitoring_distinguishes_cold_start_gaps_and_drift():
    readings = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2020-01-01 00:00", "2020-01-01 00:01", "2020-01-01 01:00"]
            ),
            "TP2": [10.0, 11.0, 12.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "anomaly_score": [np.nan, 0.2, 0.9],
            "is_scorable": [False, True, True],
            "is_alert": [False, False, True],
        }
    )
    report = monitoring_summary(
        readings,
        predictions,
        {"TP2": {"mean": 0.0, "std": 1.0}},
        ["TP2"],
        elapsed_seconds=0.3,
    )
    assert report["gap_count"] == 1 and report["scorable_count"] == 2
    assert report["alert_count"] == 1 and report["drift_review"]
    assert report["unscorable_count"] == 1
    assert report["quality_passed"]
    none = predictions.assign(is_scorable=False, anomaly_score=np.nan)
    assert not monitoring_summary(readings, none, {}, ["TP2"])["quality_passed"]


def test_unseen_state_of_constant_training_signal_requests_review():
    observations = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-07-01", periods=2, freq="min"),
            "LPS": [0.0, 1.0],
        }
    )
    predictions = pd.DataFrame(
        {"is_alert": [False, False], "is_scorable": [True, True]}
    )
    result = monitoring_summary(
        observations, predictions, {"LPS": {"mean": 0.0, "std": 0.0}}, ["LPS"]
    )
    assert result["constant_feature_change_fractions"] == {"LPS": 0.5}
    assert result["drift_review"] is True
    assert result["max_mean_shift"] == 0.0
