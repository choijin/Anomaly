"""Local checks for job windows, release gates, and controlled model promotion."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from metric_anomaly.databricks_helpers import (
    batch_window,
    parse_time,
    table_name,
    validate_identifier,
    validate_training_windows,
)
from metric_anomaly.promote_model import promote_model
from metric_anomaly.train_model import passes_validation
from metric_anomaly.detection import apply_alert_policy


class JobWindowTests(unittest.TestCase):
    def test_cli_dates_mean_utc_and_invalid_dates_are_rejected(self):
        self.assertEqual(parse_time("2026-02-01"), pd.Timestamp("2026-02-01T00:00:00Z"))
        for invalid in ["not-a-date", "NaT", ""]:
            with self.subTest(timestamp=invalid), self.assertRaises(ValueError):
                parse_time(invalid)

    def test_default_batch_uses_previous_complete_utc_day(self):
        # Local midnight must not change the processing day's UTC boundaries.
        start, end = batch_window(now=pd.Timestamp("2026-10-30T23:30:00-05:00"))
        self.assertEqual(start, pd.Timestamp("2026-10-30T00:00:00Z"))
        self.assertEqual(end, pd.Timestamp("2026-10-31T00:00:00Z"))

    def test_explicit_backfill_window_is_preserved_and_normalized_to_utc(self):
        start, end = batch_window("2026-02-01T00:00:00-06:00", "2026-02-03T00:00:00-06:00")
        self.assertEqual(start, pd.Timestamp("2026-02-01T06:00:00Z"))
        self.assertEqual(end, pd.Timestamp("2026-02-03T06:00:00Z"))
        self.assertEqual(str(start.tz), "UTC")
        self.assertEqual(str(end.tz), "UTC")

    def test_partial_empty_and_reversed_windows_are_rejected(self):
        for start, end in [
            ("2026-02-01", ""),
            ("", "2026-02-02"),
            ("2026-02-01", "2026-02-01"),
            ("2026-02-02", "2026-02-01"),
        ]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                batch_window(start, end)

    def test_training_calibration_and_evaluation_boundaries_are_strict(self):
        valid = ["2026-01-01", "2026-01-15", "2026-01-22", "2026-01-29"]
        self.assertEqual(len(validate_training_windows(*valid)), 4)
        for invalid in [
            ["2026-01-01", "2026-01-01", "2026-01-22", "2026-01-29"],
            ["2026-01-01", "2026-01-22", "2026-01-15", "2026-01-29"],
            ["2026-01-01", "2026-01-15", "2026-01-22", "2026-01-22"],
        ]:
            with self.subTest(boundaries=invalid), self.assertRaises(ValueError):
                validate_training_windows(*invalid)

    def test_sql_identifiers_allow_meaningful_names_and_reject_sql_fragments(self):
        self.assertEqual(validate_identifier("server_metrics"), "server_metrics")
        self.assertEqual(table_name("analytics", "anomaly_dev", "server_metrics"),
                         "analytics.anomaly_dev.server_metrics")
        for invalid in ["", "1table", "analytics.prod", "table; DROP TABLE assets", "a`", "a b", "a-b"]:
            with self.subTest(identifier=invalid), self.assertRaises(ValueError):
                table_name("analytics", "anomaly_dev", invalid)


class ModelValidationTests(unittest.TestCase):
    def setUp(self):
        self.metrics = {
            "events_total": 12,
            "normal_series_days": 18.5,
            "event_recall": 1.0,
            "false_alerts_per_normal_day": 0.05,
        }

    def test_held_out_event_recall_and_alert_budget_are_both_required(self):
        self.assertTrue(passes_validation(self.metrics))
        self.assertFalse(passes_validation({**self.metrics, "event_recall": 0.79}))
        self.assertFalse(passes_validation({**self.metrics, "false_alerts_per_normal_day": 0.11}))
        self.assertFalse(passes_validation({**self.metrics, "events_total": 0}))
        self.assertFalse(passes_validation({**self.metrics, "normal_series_days": 0}))
        self.assertFalse(passes_validation({**self.metrics, "event_recall": 1.1}))
        self.assertFalse(passes_validation({**self.metrics, "false_alerts_per_normal_day": -0.1}))

    def test_missing_or_nonfinite_metrics_cannot_release_a_model(self):
        for field in self.metrics:
            incomplete = {key: value for key, value in self.metrics.items() if key != field}
            with self.subTest(missing=field):
                self.assertFalse(passes_validation(incomplete))
            for invalid in [math.nan, math.inf, -math.inf]:
                with self.subTest(field=field, value=invalid):
                    self.assertFalse(passes_validation({**self.metrics, field: invalid}))

    def test_production_cannot_inspect_candidate_without_approval(self):
        from metric_anomaly.score_batch import main
        arguments = ["score_batch", "--catalog", "analytics", "--schema", "metric_anomaly_prod", "--model-alias", "candidate"]
        with patch("sys.argv", arguments), self.assertRaisesRegex(ValueError, "development-only"):
            main()


class BatchAlertBoundaryTests(unittest.TestCase):
    def test_three_previous_samples_preserve_episodes_across_batch_boundaries(self):
        # A includes an active episode whose earlier votes must be replayed.
        # B also includes a missing interval, which must reset the episode.
        metric_rows = []
        for series, offsets in [
            ("A", [0, 5, 10, 15, 20, 25, 30, 35, 40]),
            ("B", [0, 5, 10, 25, 30, 35, 40, 45, 50]),
        ]:
            metric_rows.append(pd.DataFrame({
                "series_id": series,
                "timestamp": pd.Timestamp("2026-01-01T00:00:00Z") + pd.to_timedelta(offsets, unit="min"),
                "point_flag": [True, False, True, True, True, False, False, True, True],
            }))
        readings = pd.concat(metric_rows, ignore_index=True)
        complete_history = apply_alert_policy(readings)
        for window_start in sorted(readings.timestamp.unique())[3:]:
            context_start = window_start - pd.Timedelta(minutes=15)
            replayed = apply_alert_policy(readings.loc[readings.timestamp >= context_start])
            with self.subTest(window_start=window_start):
                pd.testing.assert_frame_equal(
                    replayed.loc[replayed.timestamp >= window_start],
                    complete_history.loc[complete_history.timestamp >= window_start],
                )


class ModelPromotionTests(unittest.TestCase):
    source_model = "analytics.anomaly_dev.metric_anomaly"
    target_model = "analytics.anomaly_prod.metric_anomaly"

    def setUp(self):
        self.client = MagicMock()
        self.client.get_model_version.return_value = SimpleNamespace(
            version="4", tags={"validation_passed": "true", "data_source": "reviewed_real"},
        )
        self.client.get_model_version_by_alias.return_value = SimpleNamespace(version="7")
        self.client.copy_model_version.return_value = SimpleNamespace(version="9")

    def promote(self, **overrides):
        arguments = {
            "client": self.client,
            "source_model": self.source_model,
            "source_version": "4",
            "target_model": self.target_model,
            "approval_ticket": "OPS-123",
        }
        return promote_model(**(arguments | overrides))

    def test_promotion_copies_an_explicit_version_and_records_previous_champion(self):
        promoted = self.promote()
        self.assertEqual(str(promoted.version), "9")
        self.client.get_model_version.assert_called_with(self.source_model, "4")
        self.client.copy_model_version.assert_called_once_with(
            f"models:/{self.source_model}/4", self.target_model,
        )
        self.client.get_model_version_by_alias.assert_called_once_with(self.target_model, "champion")
        self.client.set_model_version_tag.assert_any_call(self.target_model, "9", "approval_ticket", "OPS-123")
        self.client.set_model_version_tag.assert_any_call(
            self.target_model, "9", "previous_champion_version", "7",
        )
        self.client.set_registered_model_alias.assert_called_once_with(self.target_model, "champion", "9")
        self.assertEqual(self.client.method_calls[-1][0], "set_registered_model_alias")

    def test_missing_approval_or_failed_validation_never_changes_champion(self):
        for ticket, tags in [
            ("", {"validation_passed": "true", "data_source": "reviewed_real"}),
            ("   ", {"validation_passed": "true", "data_source": "reviewed_real"}),
            ("OPS-123", {"validation_passed": "false", "data_source": "reviewed_real"}),
            ("OPS-123", {"data_source": "reviewed_real"}),
            ("OPS-123", {"validation_passed": "true"}),
        ]:
            self.client.reset_mock()
            self.client.get_model_version.return_value.tags = tags
            with self.subTest(ticket=ticket, tags=tags), self.assertRaises(ValueError):
                self.promote(approval_ticket=ticket)
            self.client.set_registered_model_alias.assert_not_called()
            self.client.copy_model_version.assert_not_called()

    def test_public_benchmark_model_requires_explicit_demo_flag_and_development_schema(self):
        self.client.get_model_version.return_value.tags["data_source"] = "public_benchmark"
        for target_model, allow_benchmark in [
            (self.source_model, False), (self.target_model, False), (self.target_model, True),
        ]:
            with self.subTest(target=target_model, allow_benchmark=allow_benchmark), self.assertRaises(ValueError):
                self.promote(target_model=target_model, allow_benchmark=allow_benchmark)
        self.client.set_registered_model_alias.assert_not_called()
        self.client.copy_model_version.assert_not_called()

        promoted = self.promote(target_model=self.source_model, allow_benchmark=True)
        self.assertEqual(str(promoted.version), "4")
        self.client.copy_model_version.assert_not_called()
        self.client.set_registered_model_alias.assert_called_once_with(self.source_model, "champion", "4")

    def test_failed_copy_leaves_existing_production_champion_unchanged(self):
        self.client.copy_model_version.side_effect = RuntimeError("Registry unavailable")
        with self.assertRaisesRegex(RuntimeError, "Registry unavailable"):
            self.promote()
        self.client.set_registered_model_alias.assert_not_called()

    def test_aliases_and_invalid_source_versions_cannot_be_promoted(self):
        for version in ["candidate", "latest", "0", "-1", "4/other"]:
            with self.subTest(version=version), self.assertRaises(ValueError):
                self.promote(source_version=version)
        self.client.copy_model_version.assert_not_called()
        self.client.set_registered_model_alias.assert_not_called()

    def test_unavailable_registry_does_not_masquerade_as_first_promotion(self):
        self.client.get_model_version_by_alias.side_effect = RuntimeError("Permission denied")
        with self.assertRaisesRegex(RuntimeError, "Permission denied"):
            self.promote()
        self.client.set_registered_model_alias.assert_not_called()

    def test_failed_audit_metadata_write_prevents_champion_change(self):
        self.client.set_model_version_tag.side_effect = RuntimeError("Metadata write denied")
        with self.assertRaisesRegex(RuntimeError, "Metadata write denied"):
            self.promote()
        self.client.set_registered_model_alias.assert_not_called()


if __name__ == "__main__":
    unittest.main()
