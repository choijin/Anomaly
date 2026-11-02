"""A small full-pipeline integration test; the real benchmark is opt-in CI."""

import json
import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from anomaly_detection.config import load_config
from anomaly_detection.data import SENSOR_COLUMNS
from anomaly_detection.pipeline import run_experiments, select_model
from anomaly_detection.predict import DetectorArtifact, MODEL_INPUT_COLUMNS
from anomaly_detection.models.isolation_forest import IsolationForestDetector
from anomaly_detection.report import publish_results


def test_validation_selection_is_deterministic_and_does_not_take_test_metrics():
    config = load_config()
    base = {
        "event_recall": 1.0,
        "unmatched_alerts_per_day": 0.2,
        "events_total": 3,
        "inference_ms_per_row": 1.0,
    }
    results = {
        name: {"validation_metrics": dict(base)}
        for name in ["if_raw", "if_temporal", "cnn_autoencoder"]
    }
    assert select_model(results, config) == "if_raw"
    results["if_raw"]["test_metrics"] = {"event_recall": 0.0}
    results["cnn_autoencoder"]["test_metrics"] = {"event_recall": 1.0}
    assert select_model(results, config) == "if_raw"
    results["if_temporal"]["validation_metrics"]["unmatched_alerts_per_day"] = 0.1
    assert select_model(results, config) == "if_temporal"


def test_packaged_contract_excludes_labels_and_handles_datetime_units(observations):
    config = load_config()
    frame = observations.assign(segment_id=0)
    detector = IsolationForestDetector({"n_estimators": 5, "n_jobs": 1}, 42).fit(
        frame[SENSOR_COLUMNS]
    )
    artifact = DetectorArtifact("if_raw", detector, config, 0.5, {})
    reference = artifact.predict(frame)
    changed = frame.assign(
        is_reported_failure=True, event_id="irrelevant", source_record_id=-100
    )
    changed.timestamp = changed.timestamp.astype("datetime64[us]")
    actual = artifact.predict(changed)
    np.testing.assert_allclose(
        actual.anomaly_score, reference.anomaly_score, equal_nan=True
    )
    np.testing.assert_array_equal(actual.is_alert, reference.is_alert)
    with pytest.raises(ValueError, match="segment_id"):
        artifact.predict(frame.drop(columns="segment_id"))


@pytest.mark.integration
def test_three_experiments_log_reload_and_select_before_test(tmp_path):
    config = load_config()
    boundaries = pd.date_range("2020-02-01", periods=5, freq="4h")
    for index, name in enumerate(["train", "normal_development", "validation", "test"]):
        config["splits"][name] = {
            "start": str(boundaries[index]),
            "end": str(boundaries[index + 1]),
        }
    config["trusted_normal_intervals"] = [
        {"start": str(boundaries[0]), "end": str(boundaries[2])}
    ]
    config["guard_band"] = "0min"
    config["failure_events"] = [
        {"event_id": "V1", "start": "2020-02-01 09:00", "end": "2020-02-01 10:00"},
        {"event_id": "T1", "start": "2020-02-01 13:00", "end": "2020-02-01 14:00"},
    ]
    config["autoencoder"].update(
        epochs=2,
        patience=1,
        training_stride=4,
        max_training_windows=40,
        max_development_windows=30,
        threads=1,
    )
    config["isolation_forest"].update(n_estimators=8, n_jobs=1, max_training_rows=200)
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(boundaries[0], periods=960, freq="min"),
            "segment_id": 0,
        }
    )
    for column in SENSOR_COLUMNS[:7]:
        frame[column] = rng.normal(size=len(frame))
    for column in SENSOR_COLUMNS[7:]:
        frame[column] = rng.integers(0, 2, size=len(frame)).astype(float)
    frame.loc[540:599, SENSOR_COLUMNS[:7]] += 5
    frame.loc[780:839, SENSOR_COLUMNS[:7]] += 5
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    output = tmp_path / "run"
    summary = run_experiments(
        frame, config, output, tracking_uri=uri, experiment_name="integration"
    )
    assert len(summary["per_model"]) == 3
    assert (
        summary["model_reload_parity"] and summary["selected_pyfunc_full_test_parity"]
    )
    assert summary["test_used_for_selection"] is False
    assert summary["release_blocked"] == (summary["test_metrics"]["events_missed"] > 0)
    frozen = json.loads((output / "selection.json").read_text())
    assert "test_metrics" not in frozen
    assert frozen["selected_model"] == summary["selected_model"]
    assert set(frozen["thresholds"]) == set(summary["per_model"])
    assert all(
        item["test_metrics"]["events_total"] == 1
        for item in summary["per_model"].values()
    )
    counts = {
        item["test_metrics"]["scored_records"] for item in summary["per_model"].values()
    }
    assert len(counts) == 1
    selected = joblib.load(output / "selected_model.joblib")
    test = frame.iloc[720:].reset_index(drop=True)
    expected = selected.predict(test)
    pyfunc = mlflow.pyfunc.load_model(summary["model_uri"])
    actual = pyfunc.predict(test[MODEL_INPUT_COLUMNS])
    np.testing.assert_allclose(
        actual.anomaly_score, expected.anomaly_score, equal_nan=True
    )
    np.testing.assert_array_equal(actual.is_alert, expected.is_alert)
    assert (output / "report.md").is_file()
    published = tmp_path / "published"
    publish_results(summary, output, published)
    portable = json.loads((published / "benchmark_results.json").read_text())
    assert portable["selected_model"] == summary["selected_model"]
    assert "model_uri" not in portable and "selected_model_path" not in portable
    assert "run_id" not in portable["per_model"][summary["selected_model"]]
    assert (published / "model_comparison.md").read_text() == (
        output / "report.md"
    ).read_text()
