"""Download the checksum-pinned public benchmark for local analysis."""

import urllib.request
import zipfile
from pathlib import Path

from anomaly_detection.data.ingestion import CSV_SHA256, file_digest

DATASET_URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"
ARCHIVE_SHA256 = "aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a"


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
