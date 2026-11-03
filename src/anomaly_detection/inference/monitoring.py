"""Input quality, drift, coverage, and latency for scored batches."""

import math
from typing import Any

import numpy as np
import pandas as pd


def monitoring_summary(
    observations: pd.DataFrame,
    predictions: pd.DataFrame,
    reference_statistics: dict[str, Any],
    sensor_columns: list[str],
    *,
    max_gap: str = "90s",
    elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    """Summarize input quality, drift, coverage, and batch cost."""
    missing = [column for column in sensor_columns if column not in observations]
    null_count = int(observations.reindex(columns=sensor_columns).isna().sum().sum())
    numeric = observations.reindex(columns=sensor_columns).to_numpy(dtype=float)
    nonfinite_count = int((~np.isfinite(numeric)).sum())
    times = pd.to_datetime(observations["timestamp"])
    boundaries = times.diff() > pd.Timedelta(max_gap)
    if "segment_id" in observations:
        boundaries |= observations.segment_id.diff().fillna(0).ne(0)
    gap_count = int(boundaries.sum())
    shifts: dict[str, float] = {}
    constant_changes: dict[str, float] = {}
    for column in sensor_columns:
        if column not in observations:
            continue
        reference = reference_statistics.get(column, {})
        if not isinstance(reference, dict):
            continue
        mean, scale = reference.get("mean"), reference.get("std")
        if mean is not None and scale == 0 and math.isfinite(float(mean)):
            constant_changes[column] = float(
                observations[column].ne(float(mean)).mean()
            )
        if (
            mean is not None
            and scale is not None
            and math.isfinite(float(mean))
            and float(scale) > 0
        ):
            shifts[column] = float(abs(observations[column].mean() - mean) / scale)
    alerts = predictions.get(
        "is_alert", pd.Series(False, index=predictions.index)
    ).fillna(False)
    scorable = predictions.get(
        "is_scorable", predictions.get("anomaly_score", pd.Series(dtype=float)).notna()
    )
    return {
        "row_count": len(observations),
        "missing_columns": missing,
        "null_count": null_count,
        "nonfinite_count": nonfinite_count,
        "gap_count": gap_count,
        "scorable_count": int(scorable.sum()),
        "unscorable_count": int((~scorable).sum()),
        "unscorable_fraction": float((~scorable).mean()) if len(scorable) else 1.0,
        "alert_count": int(alerts.sum()),
        "alert_fraction": float(alerts.mean()) if len(alerts) else 0.0,
        "elapsed_seconds": float(elapsed_seconds),
        "inference_ms_per_row": 1000 * elapsed_seconds / max(len(observations), 1),
        "max_mean_shift": max(shifts.values(), default=0.0),
        "feature_mean_shifts": shifts,
        "constant_feature_change_fractions": constant_changes,
        "drift_review": any(shift > 3.0 for shift in shifts.values())
        or any(fraction > 0 for fraction in constant_changes.values()),
        "quality_passed": not missing
        and null_count == 0
        and nonfinite_count == 0
        and bool(scorable.any()),
        "timestamp_semantics": "timezone-unspecified source clock",
    }
