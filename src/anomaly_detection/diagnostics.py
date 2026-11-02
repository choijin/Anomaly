"""Validation score-sensitivity diagnostics."""

import numpy as np
import pandas as pd

from .data import CONTINUOUS_COLUMNS


def temporal_score_sensitivity(
    detector,
    features: pd.DataFrame,
    max_rows: int = 1200,
    seed: int = 42,
) -> pd.DataFrame:
    """Permute sensor/window groups and measure fitted score changes.

    The seeded perturbations break between-group correlations. They describe
    score sensitivity, not causal importance or improvements in detection.
    """
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        raise ValueError("max_rows must be a positive integer")
    if (
        not isinstance(features, pd.DataFrame)
        or features.empty
        or features.columns.duplicated().any()
    ):
        raise ValueError("Provide a nonempty feature DataFrame with unique names")
    numeric = features.to_numpy(dtype=float)
    finite_rows = features.loc[np.isfinite(numeric).all(axis=1)]
    if finite_rows.empty:
        raise ValueError("Sensitivity diagnostics require finite validation features")
    if len(finite_rows) > max_rows:
        positions = np.linspace(0, len(finite_rows) - 1, max_rows, dtype=int)
        sample = finite_rows.iloc[positions].copy()
    else:
        sample = finite_rows.copy()
    permutation = np.random.default_rng(seed).permutation(len(sample))
    baseline = np.asarray(detector.score(sample), dtype=float)
    groups = []
    for sensor in CONTINUOUS_COLUMNS:
        columns = [
            name
            for name in sample.columns
            if name == sensor or name.startswith(f"{sensor}_")
        ]
        if columns:
            groups.append(("sensor_group", sensor, columns))
    for window in ["1min", "5min", "15min"]:
        columns = [name for name in sample.columns if name.endswith(f"_{window}")]
        if columns:
            groups.append(("rolling_window", window, columns))
    records = []
    for diagnostic_type, group, columns in groups:
        perturbed = sample.copy()
        perturbed.loc[:, columns] = sample.iloc[permutation][columns].to_numpy()
        difference = np.asarray(detector.score(perturbed), dtype=float) - baseline
        records.append(
            {
                "diagnostic_type": diagnostic_type,
                "group": group,
                "mean_absolute_score_change": float(np.abs(difference).mean()),
                "average_score_change": float(difference.mean()),
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "diagnostic_type",
            "group",
            "mean_absolute_score_change",
            "average_score_change",
        ],
    )
