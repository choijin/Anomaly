"""MLflow signatures, dependencies, and packaged inference."""

import importlib.metadata
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from anomaly_detection.data.schema import SENSOR_COLUMNS
from anomaly_detection.inference.predictor import MODEL_INPUT_COLUMNS


class AnomalyModel(mlflow.pyfunc.PythonModel):
    """Requires a prepared chronological session prefix, including continuity IDs."""

    def __init__(self, artifact):
        self.artifact = artifact
        self.reference_statistics = artifact.reference_statistics

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        return self.artifact.predict(model_input)


def log_model(artifact, example_frame: pd.DataFrame) -> str:
    """Package inference and check reload parity."""
    example = example_frame[MODEL_INPUT_COLUMNS].head(120).copy()
    example[SENSOR_COLUMNS] = example[SENSOR_COLUMNS].astype(float)
    prediction = artifact.predict(example)
    names = ["mlflow", "numpy", "pandas", "scikit-learn", "joblib", "PyYAML"]
    torch_version = importlib.metadata.version("torch").split("+")[0]
    requirements = [f"{name}=={importlib.metadata.version(name)}" for name in names]
    requirements += [
        "--extra-index-url https://download.pytorch.org/whl/cpu",
        f"torch=={torch_version}+cpu; sys_platform == 'linux' or sys_platform == 'win32'",
        f"torch=={torch_version}; sys_platform == 'darwin'",
    ]
    info = mlflow.pyfunc.log_model(
        name="detector",
        python_model=AnomalyModel(artifact),
        signature=infer_signature(example, prediction),
        input_example=example,
        code_paths=[str(Path(__file__).parents[1])],
        pip_requirements=requirements,
    )
    restored = mlflow.pyfunc.load_model(info.model_uri).predict(example)
    np.testing.assert_allclose(
        restored.anomaly_score, prediction.anomaly_score, equal_nan=True, rtol=1e-7
    )
    np.testing.assert_array_equal(restored.is_alert, prediction.is_alert)
    return info.model_uri
