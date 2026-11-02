"""Small, deterministic telemetry fixtures; CI never downloads the benchmark."""

import numpy as np
import pandas as pd
import pytest
from anomaly_detection.data import CONTINUOUS_COLUMNS, DIGITAL_COLUMNS


@pytest.fixture
def observations():
    random = np.random.default_rng(42)
    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2020-02-01", periods=180, freq="min")}
    )
    for column in CONTINUOUS_COLUMNS:
        frame[column] = random.normal(size=len(frame))
    for column in DIGITAL_COLUMNS:
        frame[column] = random.integers(0, 2, size=len(frame)).astype(float)
    return frame
