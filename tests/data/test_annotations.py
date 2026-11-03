import pandas as pd
import pytest

from anomaly_detection.data.annotations import annotate_observations, validate_intervals
from anomaly_detection.inference.alerts import apply_alert_policy


def scores(values, start="2020-04-18 00:00", frequency="1min"):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=len(values), freq=frequency),
            "anomaly_score": values,
            "segment_id": 0,
        }
    )


def event(start="2020-04-18 00:03", end="2020-04-18 00:06", identifier="F1"):
    return {
        "event_id": identifier,
        "start": start,
        "end": end,
        "description": "Air leak",
    }


def test_annotations_distinguish_unknown_and_normal_and_exclude_guard_band():
    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2020-02-01", periods=8, freq="1h")}
    )
    annotated = annotate_observations(
        frame,
        [event("2020-02-01 04:00", "2020-02-01 05:00")],
        [{"start": "2020-02-01", "end": "2020-02-01 06:00"}],
        guard_band="1h",
    )
    assert annotated.annotation_state.tolist() == ["trusted_normal"] * 4 + [
        "reported_failure",
        "trusted_normal",
        "unknown",
        "unknown",
    ]
    assert annotated.training_eligible.tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert annotated.event_id.notna().tolist() == [
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert annotated.is_reported_failure.sum() == 1


def test_interval_end_is_exclusive_and_labels_do_not_change_sensor_values():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-04-18 23:58", periods=4, freq="1min"),
            "TP2": [1.0, 2.0, 3.0, 4.0],
        }
    )
    marked = annotate_observations(
        frame, [event("2020-04-18 00:00", "2020-04-19 00:00")], []
    )
    assert marked.is_reported_failure.tolist() == [True, True, False, False]
    pd.testing.assert_series_equal(marked.TP2, frame.TP2)
    assert not marked.training_eligible.any()


def test_reject_ambiguous_overlapping_events_and_nonunique_event_ids():
    with pytest.raises(ValueError, match="Overlapping"):
        validate_intervals(
            [event(), event("2020-04-18 00:04", "2020-04-18 00:07", "F2")],
            name="events",
        )
    with pytest.raises(ValueError, match="unique"):
        validate_intervals(
            [event(), event("2020-04-18 00:07", "2020-04-18 00:08")], name="events"
        )


def test_timestamp_contract_rejects_unsorted_duplicate_or_zoned_times():
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2020-02-02", "2020-02-01"])})
    with pytest.raises(ValueError, match="sorted"):
        annotate_observations(frame, [], [])
    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2020-02-01", periods=2, freq="1h", tz="UTC")}
    )
    with pytest.raises(ValueError, match="naive"):
        annotate_observations(frame, [], [])
    duplicate = scores([1.0, 2.0])
    duplicate.loc[1, "timestamp"] = duplicate.loc[0, "timestamp"]
    with pytest.raises(ValueError, match="unique"):
        apply_alert_policy(duplicate, 1.0)
