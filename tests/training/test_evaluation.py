import numpy as np
import pandas as pd
import pytest

from anomaly_detection.inference.alerts import apply_alert_policy
from anomaly_detection.training.evaluation import (
    calibrate_threshold,
    evaluate_predictions,
)


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
