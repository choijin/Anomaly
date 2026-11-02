"""Temporal contracts that detect look-ahead, bridged gaps and label leakage."""

import numpy as np
import pandas as pd
import pytest

from anomaly_detection.data import CONTINUOUS_COLUMNS, DIGITAL_COLUMNS, SENSOR_COLUMNS
from anomaly_detection.features import build_features, sequence_endpoints


def telemetry(count=100):
    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2020-02-01", periods=count, freq="min")}
    )
    for offset, column in enumerate(CONTINUOUS_COLUMNS):
        frame[column] = np.arange(count, dtype=float) * 2 + offset
    for column in DIGITAL_COLUMNS:
        frame[column] = np.arange(count) % 2
    return frame


def test_known_rolling_statistics_and_actual_elapsed_slope():
    frame = telemetry(20)
    features = build_features(frame, ["1min", "5min"])
    assert features.loc[5, "TP2_mean_5min"] == pytest.approx(5)
    assert features.loc[5, "TP2_std_5min"] == pytest.approx(np.std([0, 2, 4, 6, 8, 10]))
    assert features.loc[5, "TP2_deviation_5min"] == pytest.approx(5)
    assert features.loc[5, "TP2_min_5min"] == 0
    assert features.loc[5, "TP2_max_5min"] == 10
    assert features.loc[5, "TP2_slope_5min"] == pytest.approx(2)
    assert features.loc[5, "TP2_difference"] == 2
    assert features.loc[1, "TP2_mean_1min"] == 1
    assert features.loc[:4, "TP2_mean_5min"].isna().all()


def test_features_are_prefix_invariant_and_exclude_labels():
    frame = telemetry()
    frame["event_id"] = "F1"
    frame["is_reported_failure"] = np.arange(len(frame)) > 50
    full = build_features(frame)
    prefix = build_features(frame.iloc[:70])
    pd.testing.assert_frame_equal(full.iloc[:70], prefix)
    changed = frame.copy()
    changed.loc[70:, CONTINUOUS_COLUMNS] = 1e9
    pd.testing.assert_frame_equal(full.iloc[:70], build_features(changed).iloc[:70])
    assert not {"timestamp", "event_id", "is_reported_failure"}.intersection(
        full.columns
    )
    assert full.columns[:15].tolist() == SENSOR_COLUMNS
    assert len(full.columns) == 148


def test_gap_resets_features_and_prevents_crossing_sequences():
    frame = telemetry(90)
    frame.loc[30:, "timestamp"] += pd.Timedelta("2h")
    features = build_features(frame)
    assert np.isnan(features.loc[30, "TP2_difference"])
    assert np.isnan(features.loc[44, "TP2_mean_15min"])
    endpoints = sequence_endpoints(frame, sequence_length=20)
    assert endpoints.tolist() == [*range(19, 30), *range(49, 90)]
    # Explicit exclusions also restart history when timestamps happen to be adjacent.
    frame["segment_id"] = (np.arange(len(frame)) >= 65).astype(int)
    endpoints = sequence_endpoints(frame, sequence_length=20)
    assert not any(65 <= endpoint < 84 for endpoint in endpoints)


def test_split_windows_have_no_shared_observations():
    frame = telemetry(100)
    first = frame.iloc[:50].reset_index(drop=True)
    second = frame.iloc[50:].reset_index(drop=True)
    train_ends = sequence_endpoints(first, 10)
    test_ends = sequence_endpoints(second, 10)
    train_times = set(first.iloc[train_ends].timestamp)
    test_times = set(second.iloc[test_ends].timestamp)
    assert not train_times.intersection(test_times)
    assert second.iloc[test_ends[0] - 9].timestamp >= frame.iloc[50].timestamp


def test_temporal_features_reject_unordered_or_duplicate_timestamps():
    frame = telemetry()
    with pytest.raises(ValueError, match="increasing"):
        sequence_endpoints(frame.iloc[::-1])
    frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    with pytest.raises(ValueError, match="increasing"):
        build_features(frame)
