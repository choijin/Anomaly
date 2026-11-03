"""Causal rolling features and gap-safe sequence endpoints.

Windows include both endpoints. Build each chronological partition separately.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from anomaly_detection.data.schema import CONTINUOUS_COLUMNS, SENSOR_COLUMNS


def segment_numbers(frame: pd.DataFrame, max_gap: str = "90s") -> np.ndarray:
    """Identify contiguous sensor runs, including explicitly supplied boundaries."""
    if "timestamp" not in frame:
        raise ValueError("timestamp is required for temporal features")
    timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
    if timestamps.isna().any() or timestamps.dt.tz is not None:
        raise ValueError("Use non-null timestamps in the source's unspecified clock")
    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise ValueError("Timestamps must be strictly increasing")
    gap = pd.Timedelta(max_gap)
    if gap <= pd.Timedelta(0):
        raise ValueError("max_gap must be positive")
    boundary = timestamps.diff().gt(gap).to_numpy()
    if "segment_id" in frame:
        if frame["segment_id"].isna().any():
            raise ValueError("segment_id must not contain nulls")
        boundary |= frame["segment_id"].ne(frame["segment_id"].shift()).to_numpy()
    if len(boundary):
        boundary[0] = True
    return np.cumsum(boundary, dtype=np.int64) - 1


def sequence_endpoints(
    frame: pd.DataFrame, sequence_length: int = 60, max_gap: str = "90s"
) -> np.ndarray:
    """Return window endpoints contained in one uninterrupted sensor segment."""
    if (
        isinstance(sequence_length, bool)
        or not isinstance(sequence_length, int)
        or sequence_length < 1
    ):
        raise ValueError("sequence_length must be a positive integer")
    segments = segment_numbers(frame, max_gap)
    if not len(frame):
        return np.empty(0, dtype=np.int64)
    positions = np.arange(len(frame))
    new_run = np.r_[True, segments[1:] != segments[:-1]]
    starts = np.maximum.accumulate(np.where(new_run, positions, 0))
    return positions[positions - starts + 1 >= sequence_length]


def build_features(
    frame: pd.DataFrame,
    rolling_windows: Sequence[str] = ("1min", "5min", "15min"),
    max_gap: str = "90s",
) -> pd.DataFrame:
    """Add rolling statistics and elapsed-minute slopes; incomplete history is NaN."""
    missing = set(SENSOR_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing sensor columns: {sorted(missing)}")
    if len(set(rolling_windows)) != len(rolling_windows):
        raise ValueError("rolling_windows must be unique")
    windows = [(str(window), pd.Timedelta(window)) for window in rolling_windows]
    if any(duration <= pd.Timedelta(0) for _, duration in windows):
        raise ValueError("Rolling durations must be positive")
    segments = segment_numbers(frame, max_gap)
    raw = frame.loc[:, SENSOR_COLUMNS].astype(float)
    if not np.isfinite(raw.to_numpy()).all():
        raise ValueError("Model features require finite sensor readings")
    names = [
        *SENSOR_COLUMNS,
        *[f"{column}_difference" for column in CONTINUOUS_COLUMNS],
    ]
    for window, _ in windows:
        for column in CONTINUOUS_COLUMNS:
            names.extend(
                f"{column}_{statistic}_{window}"
                for statistic in ["mean", "std", "deviation", "min", "max", "slope"]
            )
    output = pd.DataFrame(np.nan, index=frame.index, columns=names, dtype=float)
    output.loc[:, SENSOR_COLUMNS] = raw.to_numpy()
    for segment in np.unique(segments):
        positions = np.flatnonzero(segments == segment)
        times = pd.DatetimeIndex(frame.iloc[positions]["timestamp"])
        values = raw.iloc[positions][CONTINUOUS_COLUMNS].copy()
        values.index = times
        elapsed = pd.Series((times - times[0]).total_seconds() / 60, index=times)
        result: dict[str, np.ndarray] = {
            f"{column}_difference": values[column].diff().to_numpy()
            for column in CONTINUOUS_COLUMNS
        }
        for window, duration in windows:
            rolling = values.rolling(duration, closed="both", min_periods=1)
            means, deviations = rolling.mean(), rolling.std(ddof=0)
            minimum, maximum = rolling.min(), rolling.max()
            # Vectorized OLS with actual elapsed time; all sums share causal bounds.
            count = elapsed.rolling(duration, closed="both", min_periods=1).count()
            sum_time = elapsed.rolling(duration, closed="both", min_periods=1).sum()
            sum_time_squared = (
                elapsed.pow(2).rolling(duration, closed="both", min_periods=1).sum()
            )
            sum_value = rolling.sum()
            sum_time_value = (
                values.mul(elapsed, axis=0)
                .rolling(duration, closed="both", min_periods=1)
                .sum()
            )
            denominator = count * sum_time_squared - sum_time.pow(2)
            slope = (
                sum_time_value.mul(count, axis=0) - sum_value.mul(sum_time, axis=0)
            ).div(denominator.replace(0, np.nan), axis=0)
            complete = np.asarray(times - times[0] >= duration)
            for column in CONTINUOUS_COLUMNS:
                statistics = {
                    "mean": means[column],
                    "std": deviations[column],
                    "deviation": values[column] - means[column],
                    "min": minimum[column],
                    "max": maximum[column],
                    "slope": slope[column],
                }
                for statistic, series in statistics.items():
                    result[f"{column}_{statistic}_{window}"] = np.where(
                        complete, series.to_numpy(), np.nan
                    )
        output.iloc[positions, len(SENSOR_COLUMNS) :] = np.column_stack(
            [result[name] for name in names[len(SENSOR_COLUMNS) :]]
        )
    return output
