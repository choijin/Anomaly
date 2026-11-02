"""Databricks ingestion, training, batch scoring, and promotion.

Inference uses a bounded driver batch to preserve temporal and alert state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import pandas as pd

from .config import config_digest, load_config
from .operations import (
    monitoring_summary,
    passes_validation,
    promote_version,
    qualified_name,
    time_bounds,
)


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-driver-rows", type=int, default=500_000)
    return parser


def _spark():
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    # These timestamps are an unspecified historical source clock, not UTC events.
    spark.conf.set("spark.sql.timestampType", "TIMESTAMP_NTZ")
    return spark


def _latest_version(spark, table: str) -> int:
    return int(spark.sql(f"DESCRIBE HISTORY {table} LIMIT 1").first()["version"])


def _bounded_pandas(rows, row_limit: int) -> pd.DataFrame:
    if row_limit <= 0:
        raise ValueError("max-driver-rows must be positive")
    frame = rows.orderBy("timestamp").limit(row_limit + 1).toPandas()
    if len(frame) > row_limit:
        raise ValueError(
            "Driver row limit exceeded; review the replay range or implement persistent per-asset state"
        )
    if frame.empty:
        raise ValueError("The selected source range contains no observations")
    return frame


def _merge(spark, destination: str, updates, keys: list[str]) -> None:
    from delta.tables import DeltaTable

    if not spark.catalog.tableExists(destination):
        updates.write.format("delta").mode("errorifexists").saveAsTable(destination)
        return
    condition = " AND ".join(f"saved.{key} = incoming.{key}" for key in keys)
    (
        DeltaTable.forName(spark, destination)
        .alias("saved")
        .merge(updates.alias("incoming"), condition)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )


def _source_metadata(frame: pd.DataFrame) -> dict[str, str]:
    result = {}
    for name in ("data_source", "source_sha256"):
        if name not in frame or frame[name].isna().any() or frame[name].nunique() != 1:
            raise ValueError(f"Require one non-null {name} across the source snapshot")
        result[name] = str(frame[name].iloc[0])
    if result["data_source"] not in {"public_benchmark", "site_reviewed"}:
        raise ValueError("Unknown data provenance")
    return result


def _model_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve native-gap metadata without exposing it as a sensor predictor."""
    from .data import validate_observations
    from .predict import MODEL_INPUT_COLUMNS

    validate_observations(frame)
    if "segment_id" not in frame:
        raise ValueError("Prepared observations require native-gap segment_id metadata")
    return frame[MODEL_INPUT_COLUMNS].copy()


def load_data() -> None:
    """Load the public CSV into a development Delta table without overwriting it."""
    from .data import (
        SENSOR_COLUMNS,
        file_digest,
        load_raw,
        prepare_observations,
        validate_observations,
    )
    from pyspark.sql.types import (
        DoubleType,
        LongType,
        StringType,
        StructField,
        StructType,
        TimestampNTZType,
    )

    parser = _parser("Prepare a MetroPT-3 CSV uploaded to a Unity Catalog Volume")
    parser.add_argument(
        "--raw-path",
        required=True,
        help="/Volumes/catalog/schema/volume/MetroPT3(AirCompressor).csv",
    )
    args = parser.parse_args()
    if not args.schema.endswith("_dev"):
        raise ValueError("The public benchmark loader is development-only")
    config = load_config(args.config)
    raw_path = Path(args.raw_path)
    source_checksum = file_digest(raw_path)
    prepared = prepare_observations(load_raw(raw_path), config)
    validate_observations(prepared)
    if len(prepared) > args.max_driver_rows:
        raise ValueError("Prepared data exceeds max-driver-rows")
    prepared = _model_inputs(prepared)
    prepared["data_source"] = "public_benchmark"
    prepared["source_sha256"] = source_checksum
    schema = StructType(
        [
            StructField("timestamp", TimestampNTZType(), False),
            StructField("segment_id", LongType(), False),
        ]
        + [StructField(column, DoubleType(), False) for column in SENSOR_COLUMNS]
        + [
            StructField("data_source", StringType(), False),
            StructField("source_sha256", StringType(), False),
        ]
    )
    spark = _spark()
    table = qualified_name(args.catalog, args.schema, "sensor_observations")
    spark.createDataFrame(prepared, schema=schema).write.format("delta").mode(
        "errorifexists"
    ).saveAsTable(table)
    print(
        json.dumps(
            {
                "source_table": table,
                "source_version": _latest_version(spark, table),
                "rows": len(prepared),
                "source_sha256": source_checksum,
            }
        )
    )


def train_model() -> None:
    """Run the frozen protocol against one pinned Delta snapshot and register a candidate."""
    import mlflow
    from mlflow import MlflowClient
    from .data import validate_observations
    from .pipeline import run_experiments

    parser = _parser(
        "Compare normal-only models and register the validation-selected candidate"
    )
    parser.add_argument("--experiment-path", required=True)
    args = parser.parse_args()
    if not args.schema.endswith("_dev"):
        raise ValueError("Benchmark experimentation belongs in the development schema")
    config = load_config(args.config)
    spark = _spark()
    source_table = qualified_name(args.catalog, args.schema, "sensor_observations")
    model_name = qualified_name(args.catalog, args.schema, "compressor_anomaly")
    version = _latest_version(spark, source_table)
    source = spark.read.option("versionAsOf", version).table(source_table)
    observations = _bounded_pandas(source, args.max_driver_rows)
    metadata: dict[str, Any] = _source_metadata(observations)
    validate_observations(observations)
    metadata.update(
        {
            "source_table": source_table,
            "source_table_version": version,
            "config_digest": config_digest(config),
        }
    )
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    with tempfile.TemporaryDirectory(prefix="metropt-training-") as temporary:
        summary = run_experiments(
            _model_inputs(observations),
            config,
            Path(temporary),
            tracking_uri="databricks",
            experiment_name=args.experiment_path,
            source_metadata=metadata,
        )
        model_uri = summary.get("model_uri")
        if not model_uri:
            raise ValueError("Training did not produce a tracked selected model")
        accepted = passes_validation(summary["validation_metrics"], config["selection"])
        candidate = mlflow.register_model(model_uri, model_name)
        run_id = summary.get("selected_run_id") or candidate.run_id
        if run_id:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_input(
                    mlflow.data.from_spark(
                        source,
                        table_name=source_table,
                        version=str(version),
                        name="sensor_snapshot",
                    ),
                    context="source_snapshot",
                )
        client = MlflowClient()
        tags = {
            "validation_passed": str(accepted).lower(),
            "data_source": metadata["data_source"],
            "source_table": source_table,
            "source_table_version": str(version),
            "source_sha256": metadata["source_sha256"],
            "config_digest": config_digest(config),
            "selected_model": summary["selected_model"],
            "selection_partition": "validation",
            "review_required": "true",
            "training_run_id": str(run_id),
            "release_blocked": str(summary.get("release_blocked", True)).lower(),
            "release_block_reasons": json.dumps(
                summary.get("release_block_reasons", ["Missing release-review outcome"])
            ),
            "test_event_recall": str(summary["test_metrics"].get("event_recall")),
            "test_events_total": str(summary["test_metrics"].get("events_total")),
            "test_events_missed": str(summary["test_metrics"].get("events_missed")),
        }
        for key, value in tags.items():
            client.set_model_version_tag(model_name, str(candidate.version), key, value)
        client.set_registered_model_alias(
            model_name, "candidate", str(candidate.version)
        )
        print(
            json.dumps(
                {
                    "model_name": model_name,
                    "candidate_version": str(candidate.version),
                    "validation_passed": accepted,
                    "release_blocked": summary.get("release_blocked", True),
                    "selected_model": summary["selected_model"],
                },
                indent=2,
            )
        )


def score_batch() -> None:
    """Score a bounded replay prefix, then write only the requested half-open range."""
    import mlflow
    from mlflow import MlflowClient
    from pyspark.sql import functions as F
    from .data import SENSOR_COLUMNS, validate_observations

    parser = _parser("Score one compressor with preserved temporal and alert context")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--replay-start",
        required=True,
        help="Fixed session anchor; keep the same anchor across batches",
    )
    parser.add_argument(
        "--source-version",
        type=int,
        required=True,
        help="Explicit immutable Delta version",
    )
    parser.add_argument(
        "--model-alias", default="champion", choices=("champion", "candidate")
    )
    args = parser.parse_args()
    lower, upper = time_bounds(args.start, args.end)
    anchor, _ = time_bounds(args.replay_start, args.end)
    if anchor > lower or args.source_version < 0:
        raise ValueError(
            "Require replay-start <= start and a nonnegative source-version"
        )
    if args.model_alias == "candidate" and not args.schema.endswith("_dev"):
        raise ValueError("Only the development schema may score a candidate")
    config = load_config(args.config)
    spark = _spark()
    source_table = qualified_name(args.catalog, args.schema, "sensor_observations")
    model_name = qualified_name(args.catalog, args.schema, "compressor_anomaly")
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    candidate = MlflowClient().get_model_version_by_alias(model_name, args.model_alias)
    if args.model_alias == "champion" and (
        candidate.tags.get("validation_passed") != "true"
        or candidate.tags.get("release_blocked") != "false"
    ):
        raise ValueError(
            "Champion scoring requires an explicitly unblocked, validated release"
        )
    if not args.schema.endswith("_dev") and (
        candidate.tags.get("validation_passed") != "true"
        or candidate.tags.get("data_source") != "site_reviewed"
    ):
        raise ValueError(
            "Production scoring requires a validated, site-reviewed champion"
        )
    # Resolve the mutable alias once. A promotion during the batch cannot change it.
    model_version = str(candidate.version)
    if candidate.tags.get("config_digest") != config_digest(config):
        raise ValueError(
            "The scoring configuration differs from the registered model protocol"
        )
    source = (
        spark.read.option("versionAsOf", args.source_version)
        .table(source_table)
        .where(
            (F.col("timestamp") >= F.lit(str(anchor)).cast("timestamp_ntz"))
            & (F.col("timestamp") < F.lit(str(upper)).cast("timestamp_ntz"))
        )
    )
    observations = _bounded_pandas(source, args.max_driver_rows)
    provenance = _source_metadata(observations)
    if (
        not args.schema.endswith("_dev")
        and provenance["data_source"] != "site_reviewed"
    ):
        raise ValueError("Production inference requires site-reviewed input data")
    validate_observations(observations)
    inputs = _model_inputs(observations)
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
    started = time.perf_counter()
    predictions = model.predict(inputs)
    elapsed = time.perf_counter() - started
    if len(predictions) != len(inputs) or not pd.to_datetime(
        predictions.timestamp
    ).equals(inputs.timestamp):
        raise ValueError(
            "Packaged model must return one aligned decision record per input observation"
        )
    wanted = (inputs.timestamp >= lower) & (inputs.timestamp < upper)
    decisions = predictions.loc[wanted].reset_index(drop=True)
    selected_inputs = inputs.loc[wanted].reset_index(drop=True)
    if decisions.empty:
        raise ValueError("The requested output range contains no source records")
    if not decisions["is_scorable"].any():
        raise ValueError(
            "No scorable decisions: provide more uninterrupted history before the requested range"
        )
    batch_fields = {
        "model_name": model_name,
        "model_version": model_version,
        "source_table": source_table,
        "source_table_version": args.source_version,
        "replay_start": str(anchor),
        "start": str(lower),
        "end": str(upper),
    }
    batch_id = hashlib.sha256(
        json.dumps(batch_fields, sort_keys=True).encode()
    ).hexdigest()[:24]
    output = decisions.assign(
        **batch_fields, batch_id=batch_id, data_source=provenance["data_source"]
    )
    # batch_id includes model + source versions and range: retrying the same request is idempotent.
    _merge(
        spark,
        qualified_name(args.catalog, args.schema, "anomaly_predictions"),
        spark.createDataFrame(output),
        ["batch_id", "timestamp"],
    )
    wrapper = model.unwrap_python_model()
    reference = getattr(wrapper, "reference_statistics", {})
    monitor = monitoring_summary(
        selected_inputs,
        decisions,
        reference,
        SENSOR_COLUMNS,
        max_gap=config["data"]["max_gap"],
        elapsed_seconds=elapsed,
    )
    monitor.update(batch_fields)
    monitor["batch_id"] = batch_id
    monitor["replayed_rows"] = len(inputs)
    # Store one JSON payload to keep nested monitoring fields schema-stable.
    monitoring_row = spark.createDataFrame(
        [(batch_id, json.dumps(monitor, allow_nan=False))],
        "batch_id string, metrics_json string",
    )
    _merge(
        spark,
        qualified_name(args.catalog, args.schema, "anomaly_monitoring"),
        monitoring_row,
        ["batch_id"],
    )
    print(json.dumps(monitor, indent=2, allow_nan=False))


def promote_model() -> None:
    """Review and promote an explicit candidate version; also supports reviewed rollback."""
    import mlflow
    from mlflow import MlflowClient

    parser = argparse.ArgumentParser(description=promote_model.__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--approval-reference", required=True)
    parser.add_argument("--allow-benchmark", choices=("true", "false"), default="false")
    args = parser.parse_args()
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    promoted = promote_version(
        MlflowClient(),
        args.source_model,
        args.source_version,
        args.target_model,
        args.approval_reference,
        allow_benchmark=args.allow_benchmark == "true",
    )
    print(f"champion: models:/{args.target_model}/{promoted.version}")
