import numpy as np
import pandas as pd
import pytest

from anomaly_detection.config import load_config
from anomaly_detection.data.annotations import annotate_observations
from anomaly_detection.data.ingestion import prepare_observations
from anomaly_detection.data.schema import SENSOR_COLUMNS, validate_observations
from anomaly_detection.training.splits import chronological_splits, validate_protocol


def test_snapshot_availability_and_no_future_information(observations):
    config = load_config()
    raw = observations.copy()
    raw.timestamp += pd.Timedelta("20s")
    snapshots = prepare_observations(raw, config)
    assert (snapshots.source_timestamp <= snapshots.timestamp).all()
    assert snapshots.timestamp.iloc[0] == pd.Timestamp("2020-02-01 00:01")
    np.testing.assert_array_equal(snapshots[SENSOR_COLUMNS], raw[SENSOR_COLUMNS])
    prefix = prepare_observations(raw.iloc[:80], config)
    pd.testing.assert_frame_equal(prefix, snapshots.iloc[:80])


def test_gaps_are_not_imputed(observations):
    raw = observations.drop(index=range(20, 30)).reset_index(drop=True)
    snapshots = prepare_observations(raw, load_config())
    assert len(snapshots) == len(raw)
    assert snapshots.segment_id.nunique() == 2


def test_native_gap_is_preserved_between_adjacent_snapshot_minutes(observations):
    raw = observations.iloc[:2].copy()
    raw.timestamp = pd.to_datetime(["2020-02-01 00:00:10", "2020-02-01 00:01:50"])
    snapshots = prepare_observations(raw, load_config())
    assert snapshots.timestamp.diff().iloc[1] == pd.Timedelta("1min")
    assert snapshots.segment_id.nunique() == 2


@pytest.mark.parametrize(
    "bad", ["duplicate", "null", "infinite", "digital", "order", "timezone"]
)
def test_invalid_telemetry_is_rejected(observations, bad):
    frame = observations.copy()
    if bad == "duplicate":
        frame.loc[1, "timestamp"] = frame.timestamp.iloc[0]
    if bad == "null":
        frame.loc[0, "TP2"] = np.nan
    if bad == "infinite":
        frame.loc[0, "TP2"] = np.inf
    if bad == "digital":
        frame.loc[0, "COMP"] = 0.5
    if bad == "order":
        frame = frame.iloc[::-1]
    if bad == "timezone":
        frame.timestamp = frame.timestamp.dt.tz_localize("UTC")
    with pytest.raises(ValueError):
        validate_observations(frame)


def test_protocol_keeps_complete_test_event():
    config = load_config()
    validate_protocol(config)
    config["splits"]["test"]["start"] = "2020-07-15 16:00"
    with pytest.raises(ValueError, match="wholly"):
        validate_protocol(config)


def test_chronological_splits_and_normal_filter():
    config = load_config()
    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2020-02-01", "2020-09-01", freq="h")}
    )
    annotated = annotate_observations(
        frame,
        config["failure_events"],
        config["trusted_normal_intervals"],
        config["guard_band"],
    )
    parts = chronological_splits(annotated, config)
    for earlier, later in zip(list(parts.values()), list(parts.values())[1:]):
        assert earlier.timestamp.max() < later.timestamp.min()
        assert set(earlier.timestamp).isdisjoint(later.timestamp)
    assert parts["train"].training_eligible.all()
    assert set(parts["test"].event_id.dropna()) == {"F4"}
    assert set(parts["validation"].event_id.dropna()) == {"F1", "F2", "F3"}
