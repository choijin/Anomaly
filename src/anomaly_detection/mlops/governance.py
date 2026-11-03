"""Release checks, registry promotion, and batch monitoring."""

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd


def validate_identifier(value: str) -> str:
    """Accept simple identifiers before interpolating qualified SQL names."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Use a simple SQL identifier: {value!r}")
    return value


def qualified_name(catalog: str, schema: str, name: str) -> str:
    return ".".join(validate_identifier(part) for part in (catalog, schema, name))


def time_bounds(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return half-open bounds on the original, unspecified source clock."""
    lower, upper = pd.Timestamp(start), pd.Timestamp(end)
    if (
        pd.isna(lower)
        or pd.isna(upper)
        or lower.tzinfo
        or upper.tzinfo
        or lower >= upper
    ):
        raise ValueError("Use timezone-naive source-clock timestamps with start < end")
    return lower, upper


def passes_validation(metrics: dict[str, Any], settings: dict[str, Any]) -> bool:
    """Check measured validation recall, alert burden, and latency."""
    recall = metrics.get("event_recall")
    unmatched = metrics.get("unmatched_alerts_per_day")
    latency = metrics.get("inference_ms_per_row")
    values = (recall, unmatched, latency)
    if any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
        for x in values
    ):
        return False
    event_count = metrics.get("events_total", 0)
    if (
        isinstance(event_count, bool)
        or not isinstance(event_count, (int, float))
        or not math.isfinite(event_count)
        or event_count <= 0
    ):
        return False
    return (
        settings.get("min_validation_event_recall", 2 / 3) - 1e-9 <= recall <= 1
        and 0
        <= unmatched
        <= settings.get("max_validation_unmatched_alerts_per_day", 1.0)
        and 0 <= latency <= settings.get("max_inference_ms_per_row", 25.0)
    )


def promote_version(
    client: Any,
    source_model: str,
    source_version: str,
    target_model: str,
    approval_reference: str,
    *,
    allow_benchmark: bool = False,
) -> Any:
    """Copy a reviewed version and move the champion alias last."""
    from mlflow.exceptions import MlflowException

    for name in (source_model, target_model):
        parts = name.split(".")
        if len(parts) != 3:
            raise ValueError("Model names must use catalog.schema.model")
        qualified_name(*parts)
    if not str(source_version).isdigit() or int(source_version) <= 0:
        raise ValueError("Choose an explicit positive model version")
    if not approval_reference.strip():
        raise ValueError("A review or change reference is required")
    source = client.get_model_version(source_model, str(source_version))
    if source.tags.get("validation_passed") != "true":
        raise ValueError("Only a validated version can become champion")
    if source.tags.get("release_blocked") != "false":
        raise ValueError(
            "Release is blocked or its review outcome is missing; candidate inspection is still available"
        )
    provenance = source.tags.get("data_source")
    development_demo = allow_benchmark and target_model.split(".")[1].endswith("_dev")
    if provenance != "site_reviewed" and not (
        provenance == "public_benchmark" and development_demo
    ):
        raise ValueError(
            "Production requires site_reviewed data; public benchmark demonstrations are development-only"
        )
    if not source.tags.get("config_digest"):
        raise ValueError(
            "The source version is missing its frozen protocol config_digest"
        )
    try:
        previous = client.get_model_version_by_alias(target_model, "champion")
    except MlflowException as error:
        if error.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        previous = None
    promoted = (
        source
        if source_model == target_model
        else client.copy_model_version(
            f"models:/{source_model}/{source_version}", target_model
        )
    )
    # Preserve the scoring protocol and lineage when copying across schemas.
    protocol_fields = (
        "config_digest",
        "selected_model",
        "training_run_id",
        "source_sha256",
        "source_table",
        "source_table_version",
        "selection_partition",
        "review_required",
        "release_blocked",
        "release_block_reasons",
        "test_event_recall",
        "test_events_total",
        "test_events_missed",
    )
    tags = {key: source.tags[key] for key in protocol_fields if key in source.tags}
    tags.update(
        {
            "approval_reference": approval_reference.strip(),
            "source_model": source_model,
            "source_version": str(source_version),
            "validation_passed": "true",
            "data_source": provenance,
        }
    )
    if previous is not None:
        tags["previous_champion_version"] = str(previous.version)
    for name, value in tags.items():
        client.set_model_version_tag(target_model, str(promoted.version), name, value)
    client.set_registered_model_alias(target_model, "champion", str(promoted.version))
    return promoted
