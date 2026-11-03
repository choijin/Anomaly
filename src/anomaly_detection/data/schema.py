"""Sensor names and telemetry validation shared by training and inference."""

import numpy as np
import pandas as pd

CONTINUOUS_COLUMNS = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
]
DIGITAL_COLUMNS = [
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
]
SENSOR_COLUMNS = CONTINUOUS_COLUMNS + DIGITAL_COLUMNS


def validate_observations(frame: pd.DataFrame) -> None:
    """Reject invalid telemetry rather than silently imputing or merging observations."""
    required = ["timestamp", *SENSOR_COLUMNS]
    missing = set(required) - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"Nonempty observations require columns: {sorted(missing)}")
    timestamps = frame.timestamp
    if not pd.api.types.is_datetime64_any_dtype(timestamps) or timestamps.isna().any():
        raise ValueError("timestamp must contain parsed, non-null datetimes")
    if timestamps.dt.tz is not None:
        raise ValueError(
            "MetroPT source timezone is unspecified; use naive source-clock timestamps"
        )
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and increasing")
    for column in SENSOR_COLUMNS:
        if not pd.api.types.is_numeric_dtype(
            frame[column]
        ) or pd.api.types.is_bool_dtype(frame[column]):
            raise ValueError(f"{column} must be numeric")
    if not np.isfinite(frame[SENSOR_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Sensor observations must be finite with no missing values")
    if not frame[DIGITAL_COLUMNS].isin([0, 1]).all().all():
        raise ValueError("Digital signals must be 0 or 1")
    if "segment_id" in frame:
        segments = frame.segment_id
        if (
            not pd.api.types.is_integer_dtype(segments)
            or pd.api.types.is_bool_dtype(segments)
            or segments.isna().any()
            or segments.lt(0).any()
            or not segments.is_monotonic_increasing
        ):
            raise ValueError(
                "segment_id must contain nonnegative, nondecreasing integer identifiers"
            )
