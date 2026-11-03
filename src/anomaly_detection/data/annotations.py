"""Failure-report annotations in the source clock; intervals use [start, end)."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def validate_intervals(
    intervals: Sequence[dict[str, Any]], *, name: str = "intervals"
) -> list[dict[str, Any]]:
    """Parse, sort and validate disjoint naive-clock intervals without changing ends."""
    parsed: list[dict[str, Any]] = []
    for interval in intervals:
        start, end = pd.Timestamp(interval["start"]), pd.Timestamp(interval["end"])
        if (
            pd.isna(start)
            or pd.isna(end)
            or start.tzinfo is not None
            or end.tzinfo is not None
        ):
            raise ValueError(
                f"{name} must use valid timezone-naive source-clock timestamps"
            )
        if start >= end:
            raise ValueError(f"{name} requires start < end")
        parsed.append({**interval, "start": start, "end": end})
    parsed.sort(key=lambda item: item["start"])
    for previous, current in zip(parsed, parsed[1:]):
        if current["start"] < previous["end"]:
            raise ValueError(f"Overlapping {name} are ambiguous")
    if name == "events":
        identifiers = [item.get("event_id") for item in parsed]
        if any(not isinstance(value, str) or not value for value in identifiers) or len(
            set(identifiers)
        ) != len(identifiers):
            raise ValueError("Events require unique nonempty event_id strings")
    return parsed


def annotate_observations(
    frame: pd.DataFrame,
    events: Sequence[dict[str, Any]],
    trusted_normal_intervals: Sequence[dict[str, Any]],
    guard_band: str = "24h",
) -> pd.DataFrame:
    """Label explicit normal ranges, reported failures, and unknown periods.

    Guard bands remove training eligibility without creating new failure labels.
    """
    if "timestamp" not in frame:
        raise ValueError("A timestamp column is required")
    times = pd.to_datetime(frame["timestamp"], errors="raise")
    if times.isna().any() or getattr(times.dt, "tz", None) is not None:
        raise ValueError("Use non-null timezone-naive source-clock timestamps")
    if not times.is_monotonic_increasing or times.duplicated().any():
        raise ValueError("Timestamps must be sorted and unique before annotation")
    failures = validate_intervals(events, name="events")
    normal_ranges = validate_intervals(
        trusted_normal_intervals, name="trusted normal intervals"
    )
    guard = pd.Timedelta(guard_band)
    if pd.isna(guard) or guard < pd.Timedelta(0):
        raise ValueError("guard_band must be a nonnegative duration")
    result = frame.copy()
    result["timestamp"] = times
    result["event_id"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["is_reported_failure"] = False
    result["annotation_state"] = "unknown"
    normal = np.zeros(len(result), dtype=bool)
    for interval in normal_ranges:
        normal |= times.between(
            interval["start"], interval["end"], inclusive="left"
        ).to_numpy()
    result.loc[normal, "annotation_state"] = "trusted_normal"
    excluded = np.zeros(len(result), dtype=bool)
    for event in failures:
        inside = times.between(event["start"], event["end"], inclusive="left")
        result.loc[inside, "event_id"] = event["event_id"]
        result.loc[inside, "is_reported_failure"] = True
        result.loc[inside, "annotation_state"] = "reported_failure"
        excluded |= times.between(
            event["start"] - guard, event["end"] + guard, inclusive="left"
        ).to_numpy()
    result["training_eligible"] = normal & ~excluded
    return result
