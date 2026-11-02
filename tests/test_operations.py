"""Governance tests cover real failure modes rather than mirroring every helper."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from anomaly_detection.databricks_jobs import _model_inputs
from anomaly_detection.data import SENSOR_COLUMNS
from anomaly_detection.predict import MODEL_INPUT_COLUMNS
from anomaly_detection.operations import (
    monitoring_summary,
    passes_validation,
    promote_version,
    qualified_name,
    time_bounds,
)


class Registry:
    def __init__(
        self, *, validated=True, data_source="public_benchmark", fail_tag=False
    ):
        self.source = SimpleNamespace(
            version="2",
            tags={
                "validation_passed": str(validated).lower(),
                "data_source": data_source,
                "config_digest": "frozen-protocol-sha",
                "selected_model": "isolation_forest_engineered",
                "training_run_id": "run-123",
                "source_sha256": "source-sha",
                "source_table": "main.metropt_dev.sensor_observations",
                "source_table_version": "7",
                "selection_partition": "validation",
                "review_required": "true",
                "release_blocked": "false",
                "release_block_reasons": "[]",
                "test_event_recall": "1.0",
                "test_events_total": "1",
                "test_events_missed": "0",
            },
        )
        self.previous = SimpleNamespace(version="1")
        self.calls = []
        self.fail_tag = fail_tag

    def get_model_version(self, name, version):
        return self.source

    def get_model_version_by_alias(self, name, alias):
        return self.previous

    def copy_model_version(self, uri, name):
        self.calls.append(("copy", uri, name))
        return SimpleNamespace(version="3")

    def set_model_version_tag(self, model, version, key, value):
        if self.fail_tag:
            raise RuntimeError("audit storage unavailable")
        self.calls.append(("tag", model, version, key, value))

    def set_registered_model_alias(self, model, alias, version):
        self.calls.append(("alias", model, alias, version))


def test_gate_rejects_nonfinite_missing_events_and_excess_alerts():
    settings = {
        "min_validation_event_recall": 0.6666666667,
        "max_validation_unmatched_alerts_per_day": 1.0,
        "max_inference_ms_per_row": 25.0,
    }
    metrics = {
        "event_recall": 2 / 3,
        "unmatched_alerts_per_day": 0.5,
        "events_total": 3,
        "inference_ms_per_row": 1.0,
    }
    assert passes_validation(metrics, settings)
    assert not passes_validation(
        {key: value for key, value in metrics.items() if key != "inference_ms_per_row"},
        settings,
    )
    for key, value in [
        ("event_recall", np.nan),
        ("events_total", np.nan),
        ("events_total", 0),
        ("unmatched_alerts_per_day", 1.1),
        ("inference_ms_per_row", 26),
    ]:
        assert not passes_validation({**metrics, key: value}, settings)


def test_time_bounds_preserve_source_clock_and_half_open_contract():
    start, end = time_bounds("2020-06-15", "2020-06-16")
    assert start.tzinfo is None and end - start == pd.Timedelta("1d")
    for bounds in [
        ("2020-06-15Z", "2020-06-16"),
        ("2020-06-15 00:00+00:00", "2020-06-16 00:00+00:00"),
        ("2020-06-16", "2020-06-15"),
    ]:
        with pytest.raises(ValueError):
            time_bounds(*bounds)
    with pytest.raises(ValueError):
        qualified_name("catalog; DROP TABLE x", "metropt_dev", "model")


def test_public_benchmark_can_only_be_promoted_explicitly_in_development():
    client = Registry()
    for target, demo in [
        ("main.metropt_prod.model", True),
        ("main.metropt_dev.model", False),
    ]:
        with pytest.raises(ValueError, match="Production requires"):
            promote_version(
                client,
                "main.metropt_dev.model",
                "2",
                target,
                "review-17",
                allow_benchmark=demo,
            )
    assert not client.calls
    promote_version(
        client,
        "main.metropt_dev.model",
        "2",
        "main.metropt_dev.model",
        "review-17",
        allow_benchmark=True,
    )
    assert client.calls[-1] == ("alias", "main.metropt_dev.model", "champion", "2")
    assert any(
        call[0] == "tag" and call[3:] == ("previous_champion_version", "1")
        for call in client.calls
    )


def test_failed_validation_and_audit_failure_cannot_change_champion():
    client = Registry(validated=False)
    with pytest.raises(ValueError, match="validated"):
        promote_version(
            client,
            "main.metropt_dev.model",
            "2",
            "main.metropt_dev.model",
            "review",
            allow_benchmark=True,
        )
    assert not client.calls
    client = Registry(data_source="site_reviewed", fail_tag=True)
    with pytest.raises(RuntimeError, match="audit"):
        promote_version(
            client, "main.metropt_dev.model", "2", "main.metropt_prod.model", "review"
        )
    assert all(call[0] != "alias" for call in client.calls)


def test_production_copy_and_reviewed_rollback_change_alias_last():
    client = Registry(data_source="site_reviewed")
    promote_version(
        client, "main.metropt_dev.model", "2", "main.metropt_prod.model", "change-123"
    )
    assert client.calls[0][0] == "copy"
    assert client.calls[-1] == ("alias", "main.metropt_prod.model", "champion", "3")
    copied_tags = {call[3]: call[4] for call in client.calls if call[0] == "tag"}
    for name in [
        "config_digest",
        "selected_model",
        "training_run_id",
        "source_sha256",
        "source_table",
        "source_table_version",
        "selection_partition",
        "release_blocked",
        "release_block_reasons",
        "test_event_recall",
        "test_events_total",
        "test_events_missed",
    ]:
        assert copied_tags[name] == client.source.tags[name]
    client.calls.clear()
    client.source.version = "1"
    promote_version(
        client,
        "main.metropt_prod.model",
        "1",
        "main.metropt_prod.model",
        "rollback-124",
    )
    assert client.calls[-1] == ("alias", "main.metropt_prod.model", "champion", "1")


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


def test_missing_protocol_digest_cannot_promote_an_unusable_champion():
    client = Registry(data_source="site_reviewed")
    del client.source.tags["config_digest"]
    with pytest.raises(ValueError, match="config_digest"):
        promote_version(
            client, "main.metropt_dev.model", "2", "main.metropt_prod.model", "review"
        )
    assert not client.calls


def test_native_gap_metadata_survives_input_projection_and_is_required():
    frame = pd.DataFrame({column: [0.0, 1.0, 1.0] for column in SENSOR_COLUMNS})
    frame["timestamp"] = pd.date_range("2020-01-01", periods=3, freq="min")
    # A native gap can occur even when successive snapshot timestamps are adjacent.
    frame["segment_id"] = pd.Series([7, 8, 8], dtype="int64")
    frame["event_id"] = "evaluation-only"
    inputs = _model_inputs(frame)
    assert inputs.columns.tolist() == MODEL_INPUT_COLUMNS
    assert inputs.segment_id.tolist() == [7, 8, 8]
    report = monitoring_summary(
        inputs, pd.DataFrame({"is_scorable": [True] * 3}), {}, SENSOR_COLUMNS
    )
    assert report["gap_count"] == 1
    for invalid in [
        frame.drop(columns="segment_id"),
        frame.assign(segment_id=[0.0, 1.0, 1.0]),
        frame.assign(segment_id=[0, 1, 0]),
    ]:
        with pytest.raises(ValueError, match="segment_id"):
            _model_inputs(invalid)


@pytest.mark.parametrize("release_state", ["true", None])
def test_validation_pass_cannot_override_missed_test_event_or_missing_release_review(
    release_state,
):
    client = Registry(validated=True)
    client.source.tags["test_events_missed"] = "1"
    client.source.tags["test_event_recall"] = "0.0"
    if release_state is None:
        del client.source.tags["release_blocked"]
    else:
        client.source.tags["release_blocked"] = release_state
        client.source.tags["release_block_reasons"] = (
            '["Selected candidate missed held-out event F4"]'
        )
    with pytest.raises(ValueError, match="Release is blocked"):
        promote_version(
            client,
            "main.metropt_dev.model",
            "2",
            "main.metropt_dev.model",
            "review",
            allow_benchmark=True,
        )
    assert not client.calls
