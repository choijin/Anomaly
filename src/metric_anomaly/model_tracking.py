"""Optional MLflow packaging of the complete stateless scoring contract."""

import importlib.metadata
import os
from pathlib import Path

import mlflow
import numpy as np
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel

from .detection import FEATURE_COLUMNS, MODEL_INPUT_COLUMNS


class MetricModel(PythonModel):
    def __init__(self, detector, reference_statistics=None):
        self.detector = detector
        self.reference_statistics = reference_statistics

    def predict(self, context, model_input, params=None):
        return self.detector.predict(model_input)


def log_run(detector, splits, metrics, metadata, output: Path) -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{output / 'mlflow.db'}")
    mlflow.set_tracking_uri(tracking_uri)
    metadata["runtime_versions"]["mlflow"] = mlflow.__version__
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "server-cpu-anomaly"))
    with mlflow.start_run(run_name="server-cpu-benchmark") as tracking_run:
        metadata["mlflow_run_id"] = tracking_run.info.run_id
        metadata["mlflow_tracking_uri"] = tracking_uri
        mlflow.set_tags({"data_source": metadata["data_source"], "purpose": "interview-prototype"})
        mlflow.log_params({
            "seed": metadata["seed"], "dataset_name": metadata["dataset_name"], "method": metadata["selected_method"],
            "data_sha256": metadata["data_sha256"], "threshold_budget": 0.005,
            "alert_policy": "2 of 3 consecutive five-minute readings per series",
            "features": ",".join(FEATURE_COLUMNS),
        })
        for context, split_data in zip(["training", "calibration", "evaluation"], splits):
            dataset = mlflow.data.from_pandas(
                split_data[MODEL_INPUT_COLUMNS], source=str(output / "metric_readings.csv"), name=context,
            )
            mlflow.log_input(dataset, context=context)
        mlflow.log_metrics({
            f"{method}.{key}": float(value)
            for method, values in metrics.items()
            for key, value in values.items() if isinstance(value, (int, float))
        })
        from .train_model import passes_validation
        mlflow.set_tag("validation_passed", str(passes_validation(metrics[detector.method])).lower())
        example = splits[2][MODEL_INPUT_COLUMNS].head(5)
        expected = detector.predict(example)
        model_info = mlflow.pyfunc.log_model(
            name="detector", python_model=MetricModel(detector, {
                "mean": splits[0][MODEL_INPUT_COLUMNS].mean().to_dict(),
                "std": splits[0][MODEL_INPUT_COLUMNS].std().clip(lower=1e-6).to_dict(),
            }),
            input_example=example,
            signature=infer_signature(example, expected),
            code_paths=[str(Path(__file__).parent)],
            pip_requirements=[
                f"{name}=={importlib.metadata.version(name)}"
                for name in ["mlflow", "numpy", "pandas", "scikit-learn", "joblib"]
            ],
        )
        metadata["mlflow_model_uri"] = model_info.model_uri
        loaded = mlflow.pyfunc.load_model(model_info.model_uri)
        observed = loaded.predict(splits[2][MODEL_INPUT_COLUMNS])
        expected_all = detector.predict(splits[2])
        np.testing.assert_allclose(observed.anomaly_score, expected_all.anomaly_score)
        np.testing.assert_array_equal(observed.is_anomaly, expected_all.is_anomaly)
        metadata["mlflow_prediction_parity"] = "passed_all_test_rows"
        # Metadata must include the model URI before it is logged as an artifact.
        from .run_demo import write_json
        write_json(output / "metadata.json", metadata)
        for path in output.iterdir():
            if path.suffix in {".json", ".md", ".png", ".csv"}:
                mlflow.log_artifact(str(path))
