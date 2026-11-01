"""Shared feature contract, detectors, causal alerts, and evaluation."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from .dataset import CADENCE_MINUTES

MODEL_INPUT_COLUMNS = ["value"]
FEATURE_COLUMNS = MODEL_INPUT_COLUMNS.copy()


def validate_values(metric_data: pd.DataFrame) -> None:
    if "value" not in metric_data:
        raise ValueError("Missing numeric value column")
    if metric_data.empty:
        raise ValueError("Empty metric batch")
    if not pd.api.types.is_numeric_dtype(metric_data.value) or pd.api.types.is_bool_dtype(metric_data.value):
        raise ValueError("value must be numeric")
    if not np.isfinite(metric_data.value.to_numpy(dtype=float)).all():
        raise ValueError("Metric values must be finite")


def validate_dataset(metric_data: pd.DataFrame) -> None:
    validate_values(metric_data)
    if not {"timestamp", "series_id"}.issubset(metric_data.columns):
        raise ValueError("Dataset requires timestamp and series_id")
    if not pd.api.types.is_string_dtype(metric_data.series_id) or metric_data.series_id.astype("string").str.strip().eq("").any():
        raise ValueError("series_id must be nonempty strings")
    if metric_data[["timestamp", "series_id"]].isna().any().any():
        raise ValueError("Missing timestamp or series_id")
    if not isinstance(metric_data.timestamp.dtype, pd.DatetimeTZDtype):
        raise ValueError("Timestamps must be timezone-aware datetimes")
    if metric_data.duplicated(["series_id", "timestamp"]).any():
        raise ValueError("Duplicate series/timestamp key")
    if not all(series_data.timestamp.is_monotonic_increasing for _, series_data in metric_data.groupby("series_id")):
        raise ValueError("Timestamps must increase within each series")


def build_features(metric_data: pd.DataFrame) -> pd.DataFrame:
    validate_values(metric_data)
    return metric_data[MODEL_INPUT_COLUMNS].astype(float).copy()


class AnomalyDetector:
    """Fit on presumed background history, calibrate once, then freeze the threshold."""

    def __init__(self, method: str = "isolation_forest", seed: int = 42):
        if method not in {"isolation_forest", "robust_zscore"}:
            raise ValueError(f"Unknown method: {method}")
        self.method = method
        self.seed = seed

    def fit(self, metric_data: pd.DataFrame):
        feature_data = build_features(metric_data)
        # Calling fit invalidates a prior calibration.
        self.threshold = None
        if self.method == "isolation_forest":
            self.estimator = IsolationForest(
                n_estimators=150, max_samples=256, contamination="auto",
                random_state=self.seed, n_jobs=1,
            ).fit(feature_data)
        else:
            self.center = feature_data.median()
            robust_scale = 1.4826 * (feature_data - self.center).abs().median()
            self.scale = robust_scale.where(robust_scale > 1e-6, feature_data.std(ddof=0)).clip(lower=1e-6)
        return self

    def score(self, metric_data: pd.DataFrame) -> np.ndarray:
        feature_data = build_features(metric_data)
        if self.method == "isolation_forest":
            # sklearn's score is lower for unusual points; invert for a high-is-bad score.
            return -self.estimator.score_samples(feature_data)
        return ((feature_data - self.center).abs() / self.scale).max(axis=1).to_numpy()

    def calibrate(self, metric_data: pd.DataFrame, false_positive_budget: float = 0.005):
        if not 0 < false_positive_budget < 1:
            raise ValueError("false_positive_budget must be between 0 and 1")
        self.threshold = float(np.quantile(self.score(metric_data), 1 - false_positive_budget))
        return self

    def predict(self, metric_data: pd.DataFrame) -> pd.DataFrame:
        if getattr(self, "threshold", None) is None:
            raise ValueError("Fit and calibrate before predicting")
        scores = self.score(metric_data)
        return pd.DataFrame({
            "anomaly_score": scores, "is_anomaly": scores > self.threshold,
        }, index=metric_data.index)


def apply_alert_policy(scored: pd.DataFrame) -> pd.DataFrame:
    """Two flags in three consecutive five-minute readings, separately per series.

    Input is sorted here; a missing interval resets the rolling window. Batch
    starts have no prior state. Production micro-batches must replay three prior
    rows and discard their outputs, or carry two rows plus the prior alert state.
    """
    alert_data = scored.sort_values(["series_id", "timestamp"]).copy()
    series_groups = alert_data.groupby("series_id", sort=False)
    interval_gaps = series_groups.timestamp.diff().ne(pd.Timedelta(minutes=CADENCE_MINUTES))
    continuous_segment = interval_gaps.cumsum()
    flags = alert_data.groupby([alert_data.series_id, continuous_segment], sort=False).point_flag
    alert_data["alert_active"] = flags.transform(lambda series_flags: series_flags.rolling(3, min_periods=2).sum().ge(2))
    previous_alert = alert_data.groupby([alert_data.series_id, continuous_segment], sort=False).alert_active.shift(fill_value=False)
    alert_data["alert_started"] = alert_data.alert_active & ~previous_alert
    return alert_data


def evaluate_predictions(scored: pd.DataFrame) -> tuple[dict, list[dict]]:
    truth = scored.is_anomaly.astype(bool)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, scored.point_flag, average="binary", zero_division=0,
    )
    alerts = scored.loc[scored.alert_started]
    details = []
    for event_id, event in scored.loc[truth].groupby("event_id", sort=True):
        starts = event.loc[event.alert_started, "timestamp"]
        delay = None if starts.empty else float((starts.min() - event.timestamp.min()).total_seconds() / 60)
        details.append({"event_id": event_id, "detected": delay is not None, "delay_minutes": delay})
    delays = [event["delay_minutes"] for event in details if event["detected"]]
    normal_days = float((~truth).sum() * CADENCE_MINUTES / 1440)
    false_alerts = int((~alerts.is_anomaly).sum())
    return {
        "point_precision": float(precision), "point_recall": float(recall), "point_f1": float(f1),
        "average_precision": float(average_precision_score(truth, scored.anomaly_score)),
        "point_false_positive_rate": float(scored.loc[~truth, "point_flag"].mean()),
        "event_recall": len(delays) / len(details) if details else None,
        "events_detected": len(delays), "events_total": len(details),
        "median_detection_delay_minutes": float(np.median(delays)) if delays else None,
        "alert_episodes": len(alerts), "false_alert_episodes": false_alerts,
        "normal_series_days": normal_days,
        "false_alerts_per_normal_day": false_alerts / normal_days if normal_days else None,
    }, details


def monitoring_summary(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Transparent drift heuristic, not a statistical significance test."""
    reference_features, current_features = build_features(reference), build_features(current)
    shifts = ((current_features.mean() - reference_features.mean()).abs() / reference_features.std().clip(lower=1e-6)).to_dict()
    return {
        "schema_status": "passed",
        "rows": len(current),
        "max_event_timestamp": current.timestamp.max().isoformat(),
        "absolute_standardized_mean_shift": {feature_name: float(shift) for feature_name, shift in shifts.items()},
        "drift_review_features": [feature_name for feature_name, shift in shifts.items() if shift > 1.0],
        "drift_review_threshold": 1.0,
        "note": "Mean-shift heuristic; also inspect operating regimes, tails and daily series slices. No automatic retraining.",
    }
