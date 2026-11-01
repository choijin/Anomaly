"""Release contract: a loaded model retains preprocessing, threshold and reference."""

import importlib.metadata
import tempfile
import unittest
from pathlib import Path

import mlflow
import numpy as np
from mlflow.models import infer_signature

from metric_anomaly.detection import AnomalyDetector, MODEL_INPUT_COLUMNS, build_features
from metric_anomaly.model_tracking import MetricModel
from metric_anomaly.dataset import load_dataset


class ModelPackagingTests(unittest.TestCase):
    def test_packaged_baseline_preserves_scores_and_monitoring_reference(self):
        metric_data = load_dataset().iloc[:500]
        training_data, calibration_data, scoring_data = metric_data.iloc[:300], metric_data.iloc[300:450], metric_data.iloc[450:]
        detector = AnomalyDetector("robust_zscore").fit(training_data).calibrate(calibration_data)
        training_features = build_features(training_data)
        reference = {"mean": training_features.mean().to_dict(), "std": training_features.std().to_dict()}
        expected = detector.predict(scoring_data)
        with tempfile.TemporaryDirectory() as directory:
            previous_tracking_uri = mlflow.get_tracking_uri()
            self.addCleanup(mlflow.set_tracking_uri, previous_tracking_uri)
            mlflow.set_tracking_uri(f"sqlite:///{Path(directory) / 'tracking.db'}")
            model_path = str(Path(directory) / "model")
            mlflow.pyfunc.save_model(
                path=model_path, python_model=MetricModel(detector, reference),
                signature=infer_signature(scoring_data[MODEL_INPUT_COLUMNS], expected),
                input_example=scoring_data[MODEL_INPUT_COLUMNS].head(3),
                pip_requirements=[f"{name}=={importlib.metadata.version(name)}" for name in ["mlflow", "pandas", "numpy", "scikit-learn", "joblib"]],
            )
            loaded = mlflow.pyfunc.load_model(model_path)
            observed = loaded.predict(scoring_data[MODEL_INPUT_COLUMNS])
            np.testing.assert_allclose(observed.anomaly_score, expected.anomaly_score)
            np.testing.assert_array_equal(observed.is_anomaly, expected.is_anomaly)
            self.assertEqual(loaded.unwrap_python_model().reference_statistics, reference)
            with self.assertRaises(mlflow.exceptions.MlflowException):
                loaded.predict(scoring_data[MODEL_INPUT_COLUMNS].drop(columns="value"))


if __name__ == "__main__":
    unittest.main()
