"""Read the single, version-controlled experimental protocol."""

from pathlib import Path
from typing import Any
import hashlib
import json
import yaml


def load_config(path: str | Path = "configs/experiments.yaml") -> dict[str, Any]:
    with Path(path).open() as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise ValueError("Experiment config must be a mapping")
    return config


def config_digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
