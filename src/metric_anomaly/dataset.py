"""Load the bundled, real NAB AWS CloudWatch CPU series and published labels."""

import hashlib
import json
from importlib.resources import files

import pandas as pd

CADENCE_MINUTES = 5
DATASET_NAME = "NAB EC2 CPU utilization 5f5533"
DATA_SOURCE = "public_benchmark"
SERIES_ID = "ec2_cpu_5f5533"


def dataset_metadata() -> dict:
    return json.loads(files("metric_anomaly").joinpath("datasets/source.json").read_text())


def load_dataset() -> pd.DataFrame:
    resource = files("metric_anomaly").joinpath("datasets/server_cpu.csv")
    raw_csv = resource.read_bytes()
    if hashlib.sha256(raw_csv).hexdigest() != dataset_metadata()["raw_csv_sha256"]:
        raise ValueError("Dataset checksum mismatch")
    with resource.open("r") as csv_file:
        readings = pd.read_csv(csv_file)
    # The original timestamps are offset-free. UTC is our storage convention.
    readings["timestamp"] = pd.to_datetime(readings.timestamp, utc=True)
    readings["value"] = readings.value.astype(float)
    readings["series_id"] = SERIES_ID
    readings["is_anomaly"] = False
    readings["event_id"] = ""
    readings["data_source"] = DATA_SOURCE
    windows = json.loads(files("metric_anomaly").joinpath("datasets/anomaly_windows.json").read_text())
    for number, (start, end) in enumerate(windows, start=1):
        inside = readings.timestamp.between(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"))
        readings.loc[inside, "is_anomaly"] = True
        readings.loc[inside, "event_id"] = f"{SERIES_ID}-event-{number}"
    return readings.sort_values(["series_id", "timestamp"], ignore_index=True)


def chronological_split(readings: pd.DataFrame):
    """First three days fit; next day calibrates; remaining ten days evaluate."""
    start = readings.timestamp.min()
    training_end = start + pd.Timedelta(days=3)
    calibration_end = start + pd.Timedelta(days=4)
    return (
        readings.loc[readings.timestamp < training_end].copy(),
        readings.loc[(readings.timestamp >= training_end) & (readings.timestamp < calibration_end)].copy(),
        readings.loc[readings.timestamp >= calibration_end].copy(),
    )
