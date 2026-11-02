"""Verify algorithm semantics, feature contracts and deterministic CPU training."""

import numpy as np
import pandas as pd
import pytest
import torch

from anomaly_detection.data import CONTINUOUS_COLUMNS, DIGITAL_COLUMNS, SENSOR_COLUMNS
from anomaly_detection.models.isolation_forest import IsolationForestDetector
from anomaly_detection.models.cnn_autoencoder import (
    ConvolutionalAutoencoder,
    SequenceAutoencoder,
)


def sensor_values(count=90, seed=4):
    generator = np.random.default_rng(seed)
    frame = pd.DataFrame(
        generator.normal(size=(count, len(CONTINUOUS_COLUMNS))),
        columns=CONTINUOUS_COLUMNS,
    )
    for column in DIGITAL_COLUMNS:
        frame[column] = generator.integers(0, 2, count)
    return frame[SENSOR_COLUMNS]


def test_isolation_scores_direction_reproducibility_and_path_lengths():
    generator = np.random.default_rng(42)
    training = pd.DataFrame(
        generator.normal(0, 0.1, (300, 2)), columns=["pressure", "temperature"]
    )
    first = IsolationForestDetector({"n_estimators": 30, "n_jobs": 1}, seed=12).fit(
        training
    )
    second = IsolationForestDetector({"n_estimators": 30, "n_jobs": 1}, seed=12).fit(
        training
    )
    points = pd.DataFrame([[0, 0], [20, 20]], columns=training.columns)
    scores = first.score(points)
    assert scores[1] > scores[0]
    np.testing.assert_array_equal(scores, second.score(points))
    np.testing.assert_allclose(
        scores,
        -first.model.score_samples(
            first.scaler.transform(points.to_numpy(dtype=np.float32))
        ),
    )
    assert first.average_path_length(points)[1] < first.average_path_length(points)[0]
    with pytest.raises(ValueError, match="order"):
        first.score(points[["temperature", "pressure"]])


def test_scaler_fit_only_uses_training_and_deterministic_cap():
    training = pd.DataFrame(
        {"pressure": np.arange(100), "temperature": np.arange(100) * 2}
    )
    detector = IsolationForestDetector(
        {"n_estimators": 10, "n_jobs": 1, "max_training_rows": 20}
    ).fit(training)
    expected = training.iloc[np.linspace(0, 99, 20, dtype=int)].mean().to_numpy()
    np.testing.assert_allclose(detector.scaler.mean_, expected)
    detector.score(training + 10000)
    np.testing.assert_allclose(detector.scaler.mean_, expected)


@pytest.mark.parametrize("length", [59, 60, 61])
def test_autoencoder_reconstructs_exact_input_shape(length):
    network = ConvolutionalAutoencoder(n_features=15)
    with torch.inference_mode():
        result = network(torch.zeros(2, 15, length))
    assert result.shape == (2, 15, length)


def test_autoencoder_training_inference_and_feature_errors_are_reproducible():
    training, development = sensor_values(50), sensor_values(45, 8)
    endpoints, development_endpoints = np.arange(7, 50), np.arange(7, 45)
    parameters = {
        "epochs": 2,
        "patience": 2,
        "batch_size": 8,
        "threads": 1,
        "training_stride": 4,
        "max_training_windows": 8,
        "max_development_windows": 4,
    }
    first = SequenceAutoencoder(parameters, sequence_length=8, seed=10).fit(
        training, endpoints, development, development_endpoints
    )
    second = SequenceAutoencoder(parameters, sequence_length=8, seed=10).fit(
        training, endpoints, development, development_endpoints
    )
    scores, errors = first.reconstruction_errors(development, development_endpoints)
    assert scores.shape == (len(development_endpoints),)
    assert errors.shape == (len(development_endpoints), 15)
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(
        scores,
        (errors[:, :7].mean(axis=1) + errors[:, 7:].mean(axis=1)) / 2,
        # Torch and NumPy float32 reductions can differ by a few rounding units.
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        scores, second.score(development, development_endpoints), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        first.scaler.mean_, training[CONTINUOUS_COLUMNS].mean(), rtol=1e-5
    )
    reconstruction = first.reconstruct(development.iloc[:8].to_numpy())
    assert reconstruction.shape == (8, 15)
    assert ((reconstruction[:, 7:] >= 0) & (reconstruction[:, 7:] <= 1)).all()
    assert len(first.history) == 2
    with pytest.raises(ValueError, match="order"):
        first.score(development[SENSOR_COLUMNS[::-1]], development_endpoints)
    with pytest.raises(ValueError, match="complete windows"):
        first.score(development, np.array([3]))
