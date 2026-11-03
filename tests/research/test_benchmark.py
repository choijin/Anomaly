import hashlib
import json

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest

from anomaly_detection.config import load_config
from anomaly_detection.data.schema import SENSOR_COLUMNS
from anomaly_detection.inference.predictor import MODEL_INPUT_COLUMNS
from research.benchmark import run_benchmark
from research.report import publish_results


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
    uri = f"sqlite:///{tmp_path / 'tracking' / 'mlflow.db'}"
    output = tmp_path / "run"
    summary = run_benchmark(
        frame,
        config,
        output,
        tracking_uri=uri,
        experiment_name="integration",
        source_metadata={
            "data_source": "synthetic_test",
            "source_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(frame, index=False).values.tobytes()
            ).hexdigest(),
        },
    )
    assert summary["source"]["data_source"] == "synthetic_test"
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
    evaluation = tmp_path / "evaluation"
    publish_results(summary, output, published, evaluation)
    portable = json.loads((evaluation / "benchmark_results.json").read_text())
    assert portable["selected_model"] == summary["selected_model"]
    assert "model_uri" not in portable and "selected_model_path" not in portable
    assert "run_id" not in portable["per_model"][summary["selected_model"]]
    assert (published / "model_comparison.md").read_text() == (
        output / "report.md"
    ).read_text()
    assert (published / "figures" / "failure_zoom.png").is_file()
    assert not list(published.glob("*.json")) and not list(published.glob("*.csv"))
