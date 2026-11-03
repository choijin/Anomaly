import numpy as np
import pytest

from anomaly_detection.config import load_config
from anomaly_detection.data.schema import SENSOR_COLUMNS
from anomaly_detection.inference.predictor import DetectorArtifact
from anomaly_detection.models.isolation_forest import IsolationForestDetector


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
