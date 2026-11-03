"""Score validation and elapsed-time alert persistence/cooldown."""

import numpy as np
import pandas as pd


def parse_duration(value: str, name: str, *, allow_zero: bool = False) -> pd.Timedelta:
    result = pd.Timedelta(value)
    if (
        pd.isna(result)
        or result < pd.Timedelta(0)
        or (not allow_zero and result == pd.Timedelta(0))
    ):
        raise ValueError(
            f"{name} must be {'nonnegative' if allow_zero else 'positive'}"
        )
    return result


def validate_scores(scored: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "anomaly_score"}
    if not required.issubset(scored.columns):
        raise ValueError(
            f"Missing score columns: {sorted(required - set(scored.columns))}"
        )
    result = scored.copy()
    result["timestamp"] = pd.to_datetime(result.timestamp, errors="raise")
    if (
        result.timestamp.isna().any()
        or getattr(result.timestamp.dt, "tz", None) is not None
    ):
        raise ValueError("Use non-null timezone-naive source-clock timestamps")
    if (
        not result.timestamp.is_monotonic_increasing
        or result.timestamp.duplicated().any()
    ):
        raise ValueError("Score timestamps must be sorted and unique")
    if not pd.api.types.is_numeric_dtype(result.anomaly_score):
        raise ValueError("anomaly_score must be numeric; NaN warm-up rows are allowed")
    if np.isinf(result.anomaly_score.to_numpy(dtype=float)).any():
        raise ValueError("Infinite anomaly scores are invalid")
    return result


def apply_alert_policy(
    scored: pd.DataFrame,
    threshold: float,
    persistence: str = "2min",
    cooldown: str = "30min",
    max_gap: str = "2min",
) -> pd.DataFrame:
    """Apply elapsed-time persistence and cooldown.

    Gaps reset the exceedance streak, but retain the previous alert time.
    Each call replays a complete session prefix.
    """
    result = validate_scores(scored)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    hold = parse_duration(persistence, "persistence", allow_zero=True)
    silence = parse_duration(cooldown, "cooldown", allow_zero=True)
    gap = parse_duration(max_gap, "max_gap")
    above = result.anomaly_score.ge(float(threshold)).to_numpy()
    times = result.timestamp.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    segments = (
        result.segment_id.to_numpy()
        if "segment_id" in result
        else np.zeros(len(result), dtype=int)
    )
    alerts = np.zeros(len(result), dtype=bool)
    # Only scan threshold-exceeding observations; normal periods can be long.
    eligible_indices = np.flatnonzero(above)
    streak_start: int | None = None
    previous_index = -2
    last_alert: int | None = None
    for index in eligible_indices:
        index = int(index)
        same_run = (
            index == previous_index + 1
            and previous_index >= 0
            and segments[index] == segments[previous_index]
            and times[index] - times[previous_index] <= gap.value
        )
        if not same_run:
            streak_start = int(times[index])
        ready = int(times[index]) - int(streak_start) >= hold.value
        outside_cooldown = (
            last_alert is None or int(times[index]) - last_alert >= silence.value
        )
        if ready and outside_cooldown:
            alerts[index] = True
            last_alert = int(times[index])
        previous_index = index
    result["is_above_threshold"] = above
    result["is_alert"] = alerts
    return result
