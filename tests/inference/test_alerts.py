import numpy as np
import pandas as pd

from anomaly_detection.inference.alerts import apply_alert_policy


def scores(values, start="2020-04-18 00:00", frequency="1min"):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=len(values), freq=frequency),
            "anomaly_score": values,
            "segment_id": 0,
        }
    )


def test_persistence_means_elapsed_time_and_cooldown_limits_notifications():
    predicted = apply_alert_policy(
        scores([2.0] * 10), 1.0, persistence="2min", cooldown="4min", max_gap="90s"
    )
    assert np.flatnonzero(predicted.is_alert).tolist() == [2, 6]
    assert predicted.is_above_threshold.all()


def test_below_threshold_missing_scores_segments_and_gaps_reset_persistence():
    frame = scores([2.0, 2.0, 0.0, 2.0, np.nan, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    frame.loc[7:, "segment_id"] = 1
    frame.loc[9:, "timestamp"] += pd.Timedelta("10min")
    predicted = apply_alert_policy(
        frame, 1.0, persistence="2min", cooldown="0min", max_gap="90s"
    )
    assert not predicted.is_alert.any()
    assert not predicted.loc[4, "is_above_threshold"]


def test_gap_resets_persistence_but_does_not_erase_recent_cooldown():
    frame = scores([2.0] * 6)
    frame.loc[3:, "timestamp"] += pd.Timedelta("2min")
    predicted = apply_alert_policy(
        frame, 1.0, persistence="1min", cooldown="10min", max_gap="90s"
    )
    assert np.flatnonzero(predicted.is_alert).tolist() == [1]
