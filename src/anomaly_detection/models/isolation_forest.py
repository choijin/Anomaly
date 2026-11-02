"""Isolation Forest with explicit score direction and a frozen feature contract."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class IsolationForestDetector:
    """Normal-only forest with train-fitted scaling and fixed feature order."""

    def __init__(self, parameters: dict[str, Any] | None = None, seed: int = 42):
        self.parameters = dict(parameters or {})
        self.seed = seed
        self.feature_names: list[str] = []
        self.scaler = StandardScaler()
        self.model: IsolationForest | None = None

    def _values(self, features: pd.DataFrame, *, fitting: bool = False) -> np.ndarray:
        if (
            not isinstance(features, pd.DataFrame)
            or features.empty
            or features.columns.duplicated().any()
        ):
            raise ValueError("Supply a nonempty DataFrame with unique named features")
        if not fitting and features.columns.tolist() != self.feature_names:
            raise ValueError("Feature names and order differ from training")
        values = features.to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Features must be finite; exclude incomplete history")
        return values

    def fit(self, features: pd.DataFrame) -> "IsolationForestDetector":
        values = self._values(features, fitting=True)
        self.feature_names = features.columns.tolist()
        maximum = int(self.parameters.get("max_training_rows", len(features)))
        if maximum < 2:
            raise ValueError("max_training_rows must be at least two")
        if len(values) > maximum:
            values = values[np.linspace(0, len(values) - 1, maximum, dtype=int)]
        settings = {
            name: value
            for name, value in self.parameters.items()
            if name != "max_training_rows"
        }
        settings.setdefault("n_estimators", 200)
        settings.setdefault("max_samples", "auto")
        settings.setdefault("contamination", "auto")
        settings.setdefault("n_jobs", -1)
        settings["random_state"] = self.seed
        self.model = IsolationForest(**settings)
        self.model.fit(self.scaler.fit_transform(values))
        self.training_rows = len(values)
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Higher score means more anomalous: exactly -model.score_samples(X)."""
        if self.model is None:
            raise RuntimeError("Fit the detector before scoring")
        return -self.model.score_samples(self.scaler.transform(self._values(features)))

    def average_path_length(self, features: pd.DataFrame) -> np.ndarray:
        """Return corrected mean tree depth, including leaf occupancy."""
        return self.path_lengths_from_scores(self.score(features))

    def path_lengths_from_scores(self, scores: np.ndarray) -> np.ndarray:
        """Invert sklearn's score relation s(x) = 2**(-E[h(x)] / c(n))."""
        if self.model is None:
            raise RuntimeError("Fit the detector before scoring")
        size = self.model.max_samples_
        normalizer = (
            0.0
            if size <= 1
            else (
                1.0
                if size == 2
                else 2 * (np.log(size - 1) + np.euler_gamma) - 2 * (size - 1) / size
            )
        )
        return -np.log2(scores) * normalizer
