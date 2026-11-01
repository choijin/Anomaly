"""Behavioral checks for the feature contract, chronology, and alert semantics."""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from metric_anomaly.detection import (
    FEATURE_COLUMNS,
    AnomalyDetector,
    apply_alert_policy,
    evaluate_predictions,
    build_features,
    monitoring_summary,
    validate_dataset,
)
from metric_anomaly.dataset import chronological_split, load_dataset


def metric_rows(count=24):
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=count, freq="5min", tz="UTC"),
        "series_id": "A", "value": rng.normal(50, 2, count),
        "is_anomaly": False, "event_id": "", "data_source": "public_benchmark",
    })


def alert_rows(flags, offsets=None, series="A"):
    if offsets is None:
        offsets = np.arange(len(flags)) * 5
    return pd.DataFrame({
        "timestamp": pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(offsets, unit="min"),
        "series_id": series,
        "point_flag": flags,
    })


class DataContractTests(unittest.TestCase):
    def test_valid_batch_and_interleaved_series_are_accepted(self):
        series_a = metric_rows(6)
        series_b = series_a.assign(series_id="B")
        validate_dataset(pd.concat([series_a, series_b]).sort_values("timestamp"))

    def test_missing_required_columns_are_rejected(self):
        for column in ["timestamp", "series_id", "value"]:
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_dataset(metric_rows().drop(columns=column))

    def test_nulls_and_nonfinite_metrics_are_rejected(self):
        for column, value in [
            ("timestamp", pd.NaT), ("series_id", None),
            ("value", np.nan), ("value", np.inf),
            ("value", -np.inf),
        ]:
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                rows = metric_rows()
                rows.loc[0, column] = value
                validate_dataset(rows)

    def test_duplicate_series_timestamp_is_rejected(self):
        rows = metric_rows()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_dataset(pd.concat([rows, rows.iloc[[0]]], ignore_index=True))

    def test_string_naive_and_unordered_timestamps_are_rejected(self):
        for transformation in [
            lambda rows: rows.assign(timestamp=rows.timestamp.astype(str)),
            lambda rows: rows.assign(timestamp=rows.timestamp.dt.tz_localize(None)),
            lambda rows: rows.iloc[::-1],
        ]:
            with self.subTest(transformation=transformation), self.assertRaises(ValueError):
                validate_dataset(transformation(metric_rows()))

    def test_numeric_strings_and_boolean_values_are_rejected(self):
        for replacement in [metric_rows().value.astype(str), pd.Series([True] * 24)]:
            with self.subTest(dtype=replacement.dtype), self.assertRaises(ValueError):
                validate_dataset(metric_rows().assign(value=replacement))

    def test_empty_batch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Empty"):
            validate_dataset(metric_rows().iloc[:0])

    def test_features_exclude_labels_and_metadata(self):
        rows = metric_rows()
        result = build_features(rows)
        self.assertEqual(list(result.columns), FEATURE_COLUMNS)
        changed_labels = rows.assign(is_anomaly=True, event_id="future-fault", fault_type="failure")
        assert_frame_equal(result, build_features(changed_labels))
        np.testing.assert_allclose(result.value, rows.value)
        self.assertEqual(list(result.columns), ["value"])



class ChronologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dataset()

    def test_chronological_boundaries_and_background_training_calibration(self):
        train, calibration, test = chronological_split(self.data)
        self.assertEqual([len(part) for part in (train, calibration, test)], [864, 288, 2880])
        self.assertLess(train.timestamp.max(), calibration.timestamp.min())
        self.assertLess(calibration.timestamp.max(), test.timestamp.min())
        self.assertEqual(calibration.timestamp.min(), pd.Timestamp("2014-02-17 14:27", tz="UTC"))
        self.assertEqual(test.timestamp.min(), pd.Timestamp("2014-02-18 14:27", tz="UTC"))
        self.assertFalse(train.is_anomaly.any())
        self.assertFalse(calibration.is_anomaly.any())
        self.assertEqual(test.loc[test.is_anomaly, "event_id"].nunique(), 2)
        self.assertEqual(sum(len(part) for part in (train, calibration, test)), len(self.data))

    def test_loading_is_reproducible_and_valid(self):
        validate_dataset(self.data)
        assert_frame_equal(self.data, load_dataset())

    def test_dataset_provenance_cadence_and_published_window_boundaries(self):
        self.assertEqual(len(self.data), 4032)
        self.assertEqual(set(self.data.data_source), {"public_benchmark"})
        self.assertTrue(self.data.timestamp.diff().dropna().eq(pd.Timedelta(minutes=5)).all())
        first_event = self.data.loc[self.data.event_id.eq("ec2_cpu_5f5533-event-1")]
        self.assertEqual(first_event.timestamp.min(), pd.Timestamp("2014-02-18T16:02:00Z"))
        self.assertEqual(first_event.timestamp.max(), pd.Timestamp("2014-02-19T08:42:00Z"))


class AlertPolicyTests(unittest.TestCase):
    def test_two_of_three_and_one_start_per_episode(self):
        result = apply_alert_policy(alert_rows([True, False, True, True, False, False, True, True]))
        self.assertEqual(result.alert_active.tolist(), [False, False, True, True, True, False, False, True])
        self.assertEqual(result.alert_started.tolist(), [False, False, True, False, False, False, False, True])

    def test_gap_resets_history(self):
        result = apply_alert_policy(alert_rows(
            [True, True, True, False, True], offsets=[0, 5, 20, 25, 30],
        ))
        self.assertEqual(result.alert_active.tolist(), [False, True, False, False, True])
        self.assertEqual(result.alert_started.tolist(), [False, True, False, False, True])

    def test_other_series_never_supply_votes(self):
        # B's single positive must not combine with A's positives.
        rows = pd.concat([
            alert_rows([True, True, False], series="A"),
            alert_rows([True, False, False], series="B"),
        ], ignore_index=True).sample(frac=1, random_state=3)
        result = apply_alert_policy(rows)
        self.assertEqual(result.loc[result.series_id.eq("A"), "alert_started"].tolist(), [False, True, False])
        self.assertFalse(result.loc[result.series_id.eq("B"), "alert_active"].any())

    def test_future_rows_cannot_change_past_alerts(self):
        rows = pd.concat([
            alert_rows([False, True, True, False, True, False], series="A"),
            alert_rows([True, False, True, False, False, True], series="B"),
        ], ignore_index=True)
        full = apply_alert_policy(rows)
        for timestamp in rows.timestamp.unique():
            prefix = apply_alert_policy(rows.loc[rows.timestamp.le(timestamp)])
            assert_frame_equal(prefix, full.loc[full.timestamp.le(timestamp)])


class EvaluationTests(unittest.TestCase):
    def test_event_recall_delay_and_false_alert_exposure(self):
        series_a = alert_rows([False, False, False, True, True, False, False, False], series="A")
        series_a["is_anomaly"] = [False, False, True, True, True, False, True, True]
        series_a["event_id"] = ["", "", "event-1", "event-1", "event-1", "", "event-2", "event-2"]
        series_b = alert_rows([False, True, True, True, False, False, False, False], series="B")
        series_b["is_anomaly"] = False
        series_b["event_id"] = ""
        scored = apply_alert_policy(pd.concat([series_a, series_b], ignore_index=True))
        scored["anomaly_score"] = scored.point_flag.astype(float)
        metrics, details = evaluate_predictions(scored)
        self.assertEqual(details, [
            {"event_id": "event-1", "detected": True, "delay_minutes": 10.0},
            {"event_id": "event-2", "detected": False, "delay_minutes": None},
        ])
        self.assertEqual(metrics["event_recall"], 0.5)
        self.assertEqual(metrics["events_detected"], 1)
        self.assertEqual(metrics["events_total"], 2)
        self.assertEqual(metrics["median_detection_delay_minutes"], 10.0)
        self.assertEqual(metrics["alert_episodes"], 2)
        self.assertEqual(metrics["false_alert_episodes"], 1)
        self.assertAlmostEqual(metrics["normal_series_days"], 11 * 5 / 1440)
        self.assertAlmostEqual(metrics["false_alerts_per_normal_day"], 1440 / 55)


class MonitoringTests(unittest.TestCase):
    def test_unchanged_features_have_no_drift_even_when_labels_change(self):
        reference = metric_rows()
        unchanged = monitoring_summary(reference, reference.copy())
        relabeled = reference.assign(is_anomaly=True, fault_type="failure", event_id="future")
        self.assertEqual(unchanged, monitoring_summary(reference, relabeled))
        self.assertEqual(unchanged["drift_review_features"], [])
        self.assertTrue(all(value == 0 for value in unchanged["absolute_standardized_mean_shift"].values()))
        self.assertEqual(unchanged["rows"], len(reference))
        self.assertEqual(unchanged["max_event_timestamp"], reference.timestamp.max().isoformat())

    def test_shifted_value_requires_review_without_affecting_other_features(self):
        reference = metric_rows()
        shifted = reference.assign(value=reference.value + 2.5 * reference.value.std())
        result = monitoring_summary(reference, shifted)
        self.assertEqual(result["drift_review_features"], ["value"])
        self.assertAlmostEqual(result["absolute_standardized_mean_shift"]["value"], 2.5)
        self.assertGreater(
            result["absolute_standardized_mean_shift"]["value"],
            result["drift_review_threshold"],
        )


class AnomalyDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = metric_rows(288)
        cls.calibration = metric_rows(48).assign(value=lambda rows: rows.value + 0.2)
        cls.test = metric_rows(30)
        cls.detectors = {
            method: AnomalyDetector(method).fit(cls.training).calibrate(cls.calibration)
            for method in ("robust_zscore", "isolation_forest")
        }

    def test_joblib_roundtrip_preserves_scores_predictions_and_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            for method, detector in self.detectors.items():
                with self.subTest(method=method):
                    path = Path(directory) / f"{method}.joblib"
                    joblib.dump(detector, path)
                    loaded = joblib.load(path)
                    self.assertEqual(loaded.threshold, detector.threshold)
                    np.testing.assert_array_equal(loaded.score(self.test), detector.score(self.test))
                    assert_frame_equal(loaded.predict(self.test), detector.predict(self.test))

    def test_frozen_threshold_and_predictions_ignore_test_labels(self):
        for method, detector in self.detectors.items():
            with self.subTest(method=method):
                threshold = detector.threshold
                original = detector.predict(self.test)
                relabeled = self.test.assign(is_anomaly=True, fault_type="fault", event_id="event")
                assert_frame_equal(original, detector.predict(relabeled))
                shifted = self.test.assign(value=self.test.value + 200)
                detector.predict(shifted)
                self.assertEqual(detector.threshold, threshold)

    def test_training_and_calibration_labels_do_not_influence_model(self):
        original = self.detectors["robust_zscore"]
        relabeled = AnomalyDetector("robust_zscore").fit(
            self.training.assign(is_anomaly=True, fault_type="failure", event_id="future"),
        ).calibrate(self.calibration.assign(is_anomaly=True))
        self.assertEqual(original.threshold, relabeled.threshold)
        np.testing.assert_array_equal(original.score(self.test), relabeled.score(self.test))

    def test_refit_invalidates_calibration(self):
        detector = AnomalyDetector("robust_zscore").fit(self.training).calibrate(self.calibration)
        detector.fit(self.training)
        with self.assertRaisesRegex(ValueError, "calibrate"):
            detector.predict(self.test)

    def test_prediction_requires_calibration_and_budget_is_checked(self):
        detector = AnomalyDetector("robust_zscore").fit(self.training)
        with self.assertRaises(ValueError):
            detector.predict(self.test)
        for budget in [0, 1, -0.1, 1.1]:
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                detector.calibrate(self.calibration, budget)


if __name__ == "__main__":
    unittest.main()
