"""History-aware inference shared by local and Databricks batches."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from anomaly_detection.data.features import (
    build_features,
    segment_numbers,
    sequence_endpoints,
)
from anomaly_detection.data.schema import SENSOR_COLUMNS, validate_observations
from anomaly_detection.inference.alerts import apply_alert_policy

MODEL_INPUT_COLUMNS = ["timestamp", "segment_id", *SENSOR_COLUMNS]


@dataclass
class DetectorArtifact:
    """Detector, preprocessing, threshold, and alert policy.

    Replay a fixed session prefix before filtering requested output rows.
    Changing the replay anchor can change cooldown phase.
    """

    model_name: str
    detector: object
    config: dict
    threshold: float
    reference_statistics: dict

    def score_with_details(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(MODEL_INPUT_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing model inputs: {sorted(missing)}")
        frame = frame.reset_index(drop=True)
        validate_observations(frame)
        cadence = pd.Timedelta(self.config["data"]["expected_cadence"])
        if not frame.timestamp.dt.floor(cadence).equals(frame.timestamp):
            raise ValueError(
                "Inference expects prepared minute snapshots, not native 10-second readings"
            )
        gap = self.config["data"]["max_gap"]
        endpoints = sequence_endpoints(
            frame, self.config["features"]["sequence_length"], gap
        )
        scores = np.full(len(frame), np.nan)
        details = {}
        if self.model_name == "if_temporal":
            features = build_features(
                frame, self.config["features"]["rolling_windows"], gap
            )
            if (
                len(endpoints)
                and not np.isfinite(features.iloc[endpoints].to_numpy()).all()
            ):
                raise ValueError("Sequence warm-up must cover all rolling windows")
            if len(endpoints):
                scores[endpoints] = self.detector.score(features.iloc[endpoints])
        elif self.model_name == "if_raw":
            if len(endpoints):
                scores[endpoints] = self.detector.score(
                    frame[SENSOR_COLUMNS].iloc[endpoints]
                )
        elif self.model_name == "cnn_autoencoder":
            if len(endpoints):
                values, channel_errors = self.detector.reconstruction_errors(
                    frame[SENSOR_COLUMNS], endpoints
                )
                scores[endpoints] = values
                for index, column in enumerate(SENSOR_COLUMNS):
                    errors = np.full(len(frame), np.nan)
                    errors[endpoints] = channel_errors[:, index]
                    details[f"error_{column}"] = errors
        else:
            raise ValueError(f"Unknown detector {self.model_name}")
        if self.model_name.startswith("if_") and self.detector.model is not None:
            details["average_path_length"] = self.detector.path_lengths_from_scores(
                scores
            )
        result = pd.DataFrame(
            {
                "timestamp": frame.timestamp,
                "segment_id": segment_numbers(frame, gap),
                "anomaly_score": scores,
                **details,
            }
        )
        policy = {
            key: self.config["alerts"][key]
            for key in ["persistence", "cooldown", "max_gap"]
        }
        result = apply_alert_policy(result, self.threshold, **policy)
        result["is_scorable"] = np.isfinite(scores)
        return result

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = self.score_with_details(frame)
        return result[
            [
                "timestamp",
                "anomaly_score",
                "is_above_threshold",
                "is_alert",
                "is_scorable",
            ]
        ]


def predict_file(
    model_path: str | Path, input_path: str | Path, output_path: str | Path
) -> None:
    """Load a trusted local artifact and replay prepared minute sensor history."""
    path = Path(input_path)
    frame = (
        pd.read_parquet(path)
        if path.suffix == ".parquet"
        else pd.read_csv(path, parse_dates=["timestamp"])
    )
    artifact = joblib.load(model_path)
    prediction = artifact.predict(frame)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".parquet":
        prediction.to_parquet(output, index=False)
    else:
        prediction.to_csv(output, index=False)
