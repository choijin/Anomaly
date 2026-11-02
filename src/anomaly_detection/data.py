"""Verified MetroPT-3 ingestion and causal minute snapshots (never interpolated)."""

from pathlib import Path
import hashlib
import urllib.request
import zipfile
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
DATASET_URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"
ARCHIVE_SHA256 = "aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a"
CSV_SHA256 = "db30ccb4ea402e3c8bf2c99db06e288d4f2a772f6928f9dbe26a920d69793e24"


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(folder: str | Path = "data/raw") -> Path:
    """Download the original archive, verify pinned hashes, and safely extract CSV/PDF."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "metropt3.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".download")
        urllib.request.urlretrieve(DATASET_URL, temporary)
        if file_digest(temporary) != ARCHIVE_SHA256:
            temporary.unlink()
            raise ValueError(
                "UCI archive checksum changed; review the source before updating the pin"
            )
        temporary.replace(archive)
    if file_digest(archive) != ARCHIVE_SHA256:
        raise ValueError("UCI archive checksum mismatch")
    with zipfile.ZipFile(archive) as bundle:
        expected = {"MetroPT3(AirCompressor).csv", "Data Description_Metro.pdf"}
        if set(bundle.namelist()) != expected:
            raise ValueError("Unexpected archive contents")
        bundle.extractall(folder)
    csv_path = folder / "MetroPT3(AirCompressor).csv"
    if file_digest(csv_path) != CSV_SHA256:
        raise ValueError("MetroPT-3 CSV checksum mismatch")
    return csv_path


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
