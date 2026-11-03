"""CSV ingestion, checksums, and causal minute snapshots."""

import hashlib
from pathlib import Path

import pandas as pd

from anomaly_detection.data.schema import SENSOR_COLUMNS, validate_observations

CSV_SHA256 = "db30ccb4ea402e3c8bf2c99db06e288d4f2a772f6928f9dbe26a920d69793e24"


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(path: str | Path, verify_checksum: bool = True) -> pd.DataFrame:
    if verify_checksum and file_digest(path) != CSV_SHA256:
        raise ValueError("MetroPT-3 CSV checksum mismatch")
    frame = pd.read_csv(path)
    frame = frame.rename(columns={"Unnamed: 0": "source_record_id"})
    frame["timestamp"] = pd.to_datetime(frame.timestamp, errors="raise")
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    validate_observations(frame)
    return frame


def prepare_observations(raw_frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Take the LAST observed row in each right-closed minute, available by that minute.

    No forward fill. Empty buckets are removed and become explicit sequence breaks.
    All 15 sensor values come from the same raw observation. The timestamp is the
    snapshot availability time, while source_timestamp retains its measurement time.
    """
    validate_observations(raw_frame)
    source = raw_frame[["timestamp", *SENSOR_COLUMNS]].copy()
    source["source_timestamp"] = source.timestamp
    source["native_segment"] = (
        source.timestamp.diff().gt(pd.Timedelta(config["data"]["max_gap"])).cumsum()
    )
    snapshots = (
        source.set_index("timestamp")
        .resample(config["data"]["resample_rule"], closed="right", label="right")
        .last()
        .dropna(subset=["source_timestamp"])
        .reset_index()
    )
    new_segment = snapshots.timestamp.diff().gt(
        pd.Timedelta(config["data"]["max_gap"])
    ) | snapshots.native_segment.diff().fillna(0).ne(0)
    snapshots["segment_id"] = new_segment.cumsum().astype("int64")
    snapshots = snapshots.drop(columns="native_segment")
    validate_observations(snapshots)
    return snapshots
