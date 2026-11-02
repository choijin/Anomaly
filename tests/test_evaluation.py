"""Tests for weak annotations and the operational meaning of an alert."""

import numpy as np
import pandas as pd
import pytest

from anomaly_detection.evaluation import (
    apply_alert_policy,
    calibrate_threshold,
    evaluate_predictions,
)
from anomaly_detection.labels import annotate_observations, validate_intervals


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


def test_event_detection_delay_and_pre_failure_warning_are_separate():
    frame = scores([0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    predicted = apply_alert_policy(frame, 1.0, persistence="0min", cooldown="0min")
    metrics = evaluate_predictions(predicted, [event()], pre_failure_horizon="2min")
    assert metrics["event_recall"] == 1.0
    assert metrics["events_detected"] == 1
    assert metrics["mean_detection_delay_minutes"] == 1.0
    assert metrics["pre_failure_event_recall"] == 1.0
    assert metrics["events"][0]["warning_lead_minutes"] == 2.0
    assert metrics["unmatched_alerts"] == 1
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["false_alerts_per_day"] == metrics["unmatched_alerts_per_day"]


def test_pre_failure_alert_alone_is_not_a_detected_failure():
    predicted = apply_alert_policy(
        scores([0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 1.0, persistence="0min"
    )
    metrics = evaluate_predictions(predicted, [event()], pre_failure_horizon="2min")
    assert metrics["event_recall"] == 0.0
    assert metrics["pre_failure_event_recall"] == 1.0
    assert metrics["mean_detection_delay_minutes"] is None
    assert metrics["events_missed"] == 1


def test_gaps_do_not_inflate_monitored_exposure_or_hide_unmatched_alerts():
    frame = scores([2.0, 0.0, 0.0, np.nan])
    frame.loc[2:, "timestamp"] += pd.Timedelta("1D")
    predicted = apply_alert_policy(frame, 1.0, persistence="0min")
    metrics = evaluate_predictions(predicted, [], expected_cadence="1min")
    assert metrics["monitoring_days"] == pytest.approx(3 / 1440)
    assert metrics["unmatched_alerts_per_day"] == pytest.approx(480.0)
    assert metrics["unscored_records"] == 1
    assert metrics["event_recall"] is None
    assert metrics["pr_auc"] is None
    assert metrics["recall"] is None


def test_event_without_sensor_coverage_is_reported_as_missed():
    frame = scores([0.0, 0.0, 0.0])
    predicted = apply_alert_policy(frame, 1.0, persistence="0min")
    metrics = evaluate_predictions(
        predicted, [event()], period_start="2020-04-18", period_end="2020-04-19"
    )
    assert metrics["events_total"] == 1
    assert metrics["events_missed"] == 1
    assert metrics["events"][0]["scored_records"] == 0
    assert metrics["events"][0]["observed_fraction"] == 0.0


def test_calibration_respects_budget_and_freezes_a_finite_threshold():
    frame = scores([0.0] * 3 + [10.0] * 3 + [0.0] * 4)
    calibration = calibrate_threshold(
        frame,
        [event()],
        false_alert_budget_per_day=0.0,
        persistence="0min",
        cooldown="0min",
    )
    assert np.isfinite(calibration["threshold"])
    assert calibration["metrics"]["unmatched_alerts_per_day"] == 0.0
    assert calibration["metrics"]["event_recall"] == 1.0
    frozen = calibration["threshold"]
    test = scores([100.0, 100.0, 100.0], start="2020-07-15")
    apply_alert_policy(test, frozen, **calibration["policy"])
    assert calibration["threshold"] == frozen


def test_silent_candidate_handles_tied_normal_scores_without_infinite_threshold():
    calibration = calibrate_threshold(
        scores([1.0] * 10),
        [],
        false_alert_budget_per_day=0.0,
        persistence="0min",
        cooldown="0min",
    )
    assert calibration["threshold"] > 1.0
    assert np.isfinite(calibration["threshold"])
    assert calibration["metrics"]["alerts_total"] == 0


def test_threshold_calibration_rejects_missing_scores_or_no_budget_denominator():
    with pytest.raises(ValueError, match="finite"):
        calibrate_threshold(scores([np.nan] * 4), [])
    with pytest.raises(ValueError, match="outside"):
        calibrate_threshold(scores([1.0] * 4), [event("2020-04-18", "2020-04-19")])
    with pytest.raises(ValueError, match="nonnegative"):
        calibrate_threshold(scores([1.0] * 4), [], false_alert_budget_per_day=-1.0)


def test_pr_auc_and_average_precision_are_explicitly_distinct_summaries():
    # Scores rank the negative ahead of the positive: AP=1/2 but trapezoid area=1/4.
    frame = scores([2.0, 1.0])
    predicted = apply_alert_policy(frame, 1.5, persistence="0min", cooldown="0min")
    metrics = evaluate_predictions(
        predicted, [event("2020-04-18 00:01", "2020-04-18 00:02")]
    )
    assert metrics["pr_auc"] == pytest.approx(0.25)
    assert metrics["average_precision"] == pytest.approx(0.5)
    assert metrics["pr_auc_method"] == "trapezoidal_precision_recall_curve"
