"""Shared alert policy and event metrics.

Reports are incomplete: unmatched alerts and row labels are evaluation proxies.
Point metrics use threshold exceedances; event metrics use persisted alerts.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from anomaly_detection.data.annotations import validate_intervals
from anomaly_detection.inference.alerts import (
    apply_alert_policy,
    parse_duration,
    validate_scores,
)


def _period_events(
    events: Sequence[dict[str, Any]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[dict[str, Any]]:
    return [
        event
        for event in validate_intervals(events, name="events")
        if event["start"] < end and event["end"] > start
    ]


def evaluate_predictions(
    scored: pd.DataFrame,
    events: Sequence[dict[str, Any]],
    expected_cadence: str = "1min",
    pre_failure_horizon: str = "6h",
    period_start: str | pd.Timestamp | None = None,
    period_end: str | pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Measure event detection and proxy row metrics.

    Exposure counts scored decisions, excluding missing time. Explicit period
    bounds keep unobserved events in the denominator as misses.
    """
    frame = validate_scores(scored)
    if "is_alert" not in frame or "is_above_threshold" not in frame:
        raise ValueError("Apply the shared alert policy before evaluation")
    cadence = parse_duration(expected_cadence, "expected_cadence")
    horizon = parse_duration(
        pre_failure_horizon, "pre_failure_horizon", allow_zero=True
    )
    if frame.empty:
        raise ValueError("Evaluation needs at least one observation")
    start = (
        pd.Timestamp(period_start)
        if period_start is not None
        else frame.timestamp.iloc[0]
    )
    end = (
        pd.Timestamp(period_end)
        if period_end is not None
        else frame.timestamp.iloc[-1] + cadence
    )
    if start.tzinfo is not None or end.tzinfo is not None or start >= end:
        raise ValueError("Use increasing naive-clock evaluation bounds")
    frame = frame.loc[frame.timestamp.ge(start) & frame.timestamp.lt(end)].copy()
    if frame.empty:
        raise ValueError("No observations within evaluation bounds")
    relevant_events = _period_events(events, start, end)
    finite = frame.anomaly_score.notna()
    proxy_labels = np.zeros(len(frame), dtype=bool)
    for event in relevant_events:
        proxy_labels |= frame.timestamp.between(
            event["start"], event["end"], inclusive="left"
        ).to_numpy()
    alerts = frame.is_alert.fillna(False).astype(bool).to_numpy() & finite.to_numpy()
    point_flags = frame.is_above_threshold.fillna(False).astype(bool).to_numpy()
    finite_mask = finite.to_numpy()
    negative_rows = int((finite_mask & ~proxy_labels).sum())
    monitored_days = int(finite_mask.sum()) * cadence.total_seconds() / 86400
    outside_event_days = negative_rows * cadence.total_seconds() / 86400
    unmatched = int((alerts & ~proxy_labels).sum())
    unmatched_rate = unmatched / outside_event_days if outside_event_days > 0 else None
    event_details: list[dict[str, Any]] = []
    for event in relevant_events:
        within = frame.timestamp.between(
            event["start"], event["end"], inclusive="left"
        ).to_numpy()
        valid_times = frame.loc[within & alerts, "timestamp"]
        before = frame.timestamp.between(
            event["start"] - horizon, event["start"], inclusive="left"
        ).to_numpy()
        warning_times = frame.loc[before & alerts, "timestamp"]
        first_detection = valid_times.iloc[0] if not valid_times.empty else None
        first_warning = warning_times.iloc[0] if not warning_times.empty else None
        expected_rows = max(
            1.0,
            (event["end"] - event["start"]).total_seconds() / cadence.total_seconds(),
        )
        observed_rows = int((within & finite_mask).sum())
        event_details.append(
            {
                "event_id": event["event_id"],
                "start": event["start"].isoformat(),
                "end": event["end"].isoformat(),
                "detected": first_detection is not None,
                "first_alert": first_detection.isoformat()
                if first_detection is not None
                else None,
                "detection_delay_minutes": (
                    (first_detection - event["start"]).total_seconds() / 60
                    if first_detection is not None
                    else None
                ),
                "scored_records": observed_rows,
                "observed_fraction": min(1.0, observed_rows / expected_rows),
                "pre_failure_alert": first_warning is not None,
                "pre_failure_scored_records": int((before & finite_mask).sum()),
                "pre_failure_observed_fraction": (
                    min(
                        1.0,
                        int((before & finite_mask).sum())
                        * cadence.total_seconds()
                        / horizon.total_seconds(),
                    )
                    if horizon > pd.Timedelta(0)
                    else None
                ),
                "warning_lead_minutes": (
                    (event["start"] - first_warning).total_seconds() / 60
                    if first_warning is not None
                    else None
                ),
                "partial_partition_event": event["start"] < start or event["end"] > end,
            }
        )
    detected = sum(item["detected"] for item in event_details)
    delays = [
        item["detection_delay_minutes"] for item in event_details if item["detected"]
    ]
    labels = proxy_labels[finite_mask]
    predicted = point_flags[finite_mask]
    scores = frame.loc[finite, "anomaly_score"].to_numpy(dtype=float)
    has_both_labels = labels.any() and (~labels).any()
    if has_both_labels:
        curve_precision, curve_recall, _ = precision_recall_curve(labels, scores)
        pr_auc = float(auc(curve_recall, curve_precision))
        average_precision = float(average_precision_score(labels, scores))
    else:
        pr_auc = average_precision = None
    return {
        "event_recall": detected / len(event_details) if event_details else None,
        "events_total": len(event_details),
        "events_detected": detected,
        "events_missed": len(event_details) - detected,
        "unmatched_alerts_per_day": unmatched_rate,
        "false_alerts_per_day": unmatched_rate,
        "unmatched_alerts": unmatched,
        "alerts_total": int(alerts.sum()),
        "monitoring_days": monitored_days,
        "outside_reported_event_days": outside_event_days,
        "mean_detection_delay_minutes": float(np.mean(delays)) if delays else None,
        "median_detection_delay_minutes": float(np.median(delays)) if delays else None,
        "pre_failure_event_recall": (
            sum(item["pre_failure_alert"] for item in event_details)
            / len(event_details)
            if event_details
            else None
        ),
        "pr_auc": pr_auc,
        "average_precision": average_precision,
        "pr_auc_method": "trapezoidal_precision_recall_curve",
        "precision": float(precision_score(labels, predicted, zero_division=0))
        if len(labels)
        else None,
        "recall": float(recall_score(labels, predicted, zero_division=0))
        if labels.any()
        else None,
        "scored_records": len(scores),
        "unscored_records": int((~finite_mask).sum()),
        "failure_proxy_records": int(labels.sum()),
        "events": event_details,
        "label_limitations": "Outside reported failures is unknown; false-alert and row metrics use incomplete proxy labels.",
    }


def calibrate_threshold(
    validation_scores: pd.DataFrame,
    events: Sequence[dict[str, Any]],
    *,
    false_alert_budget_per_day: float = 1.0,
    persistence: str = "2min",
    cooldown: str = "30min",
    max_gap: str = "2min",
    expected_cadence: str = "1min",
    pre_failure_horizon: str = "6h",
    period_start: str | pd.Timestamp | None = None,
    period_end: str | pd.Timestamp | None = None,
    quantiles: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Search validation quantiles under an unmatched-alert ceiling.

    Prefer event recall, then fewer unmatched alerts, then a higher threshold.
    Include a silent threshold; alert counts need not vary monotonically.
    """
    frame = validate_scores(validation_scores)
    if not np.isfinite(false_alert_budget_per_day) or false_alert_budget_per_day < 0:
        raise ValueError("false_alert_budget_per_day must be finite and nonnegative")
    finite = frame.anomaly_score.dropna().to_numpy(dtype=float)
    if len(finite) == 0:
        raise ValueError("Validation needs finite anomaly scores")
    grid = np.asarray(
        quantiles
        if quantiles is not None
        else [
            0.80,
            0.85,
            0.90,
            0.925,
            0.95,
            0.96,
            0.97,
            0.975,
            0.98,
            0.985,
            0.99,
            0.9925,
            0.995,
            0.9975,
            0.999,
            0.9995,
            0.9999,
            1.0,
        ],
        dtype=float,
    )
    if (
        grid.ndim != 1
        or len(grid) == 0
        or not np.isfinite(grid).all()
        or (grid < 0).any()
        or (grid > 1).any()
    ):
        raise ValueError("quantiles must be a nonempty finite sequence in [0, 1]")
    thresholds = np.unique(
        np.append(np.quantile(finite, grid), np.nextafter(finite.max(), np.inf))
    )
    if not np.isfinite(thresholds).all():
        raise ValueError("Scores are too large to construct a finite silent threshold")
    policy = {"persistence": persistence, "cooldown": cooldown, "max_gap": max_gap}
    candidates: list[dict[str, Any]] = []
    for threshold in thresholds:
        predicted = apply_alert_policy(frame, float(threshold), **policy)
        metrics = evaluate_predictions(
            predicted,
            events,
            expected_cadence=expected_cadence,
            pre_failure_horizon=pre_failure_horizon,
            period_start=period_start,
            period_end=period_end,
        )
        rate = metrics["unmatched_alerts_per_day"]
        if rate is not None and rate <= false_alert_budget_per_day + 1e-12:
            candidates.append({"threshold": float(threshold), "metrics": metrics})
    if not candidates:
        raise ValueError(
            "Validation requires scored observations outside reported events for alert-budget calibration"
        )
    selected = max(
        candidates,
        key=lambda item: (
            item["metrics"]["event_recall"] or 0.0,
            -item["metrics"]["unmatched_alerts_per_day"],
            item["threshold"],
        ),
    )
    return {
        **selected,
        "policy": policy,
        "method": "validation_quantile_grid_at_matched_unmatched_alert_budget",
        "false_alert_budget_per_day": float(false_alert_budget_per_day),
        "candidate_thresholds": thresholds.tolist(),
        "feasible_candidates": len(candidates),
    }
