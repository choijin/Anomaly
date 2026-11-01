"""Small shared contracts for Databricks jobs; Spark is imported only at runtime."""

import argparse
import re

import pandas as pd


def validate_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Use a simple SQL identifier: {value!r}")
    return value


def table_name(catalog: str, schema: str, name: str) -> str:
    return ".".join(validate_identifier(part) for part in (catalog, schema, name))


def parse_time(value) -> pd.Timestamp:
    """CLI dates without an offset mean UTC; explicit offsets are converted."""
    if value is None or str(value).strip() == "":
        raise ValueError("Provide a UTC date or timestamp")
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Timestamp cannot be NaT")
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def batch_window(start_time: str = "", end_time: str = "", now=None) -> tuple[pd.Timestamp, pd.Timestamp]:
    if bool(start_time) != bool(end_time):
        raise ValueError("Provide both start-time and end-time, or neither")
    if start_time:
        start, end = parse_time(start_time), parse_time(end_time)
    else:
        end = parse_time(now if now is not None else pd.Timestamp.now(tz="UTC")).normalize()
        start = end - pd.Timedelta(days=1)
    if start >= end:
        raise ValueError("start-time must be before end-time")
    return start, end


def validate_training_windows(train_start, train_end, calibration_end, evaluation_end):
    boundaries = tuple(parse_time(value) for value in (train_start, train_end, calibration_end, evaluation_end))
    if not all(earlier < later for earlier, later in zip(boundaries, boundaries[1:])):
        raise ValueError("Training, calibration and evaluation boundaries must increase")
    return boundaries


def job_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    return parser


def spark_session():
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    return spark


def latest_table_version(spark, source_table: str) -> int:
    return int(spark.sql(f"DESCRIBE HISTORY {source_table} LIMIT 1").first()["version"])


def merge_delta_table(spark, destination: str, updates, keys: list[str]) -> None:
    """Upsert without duplicate retry rows. Jobs run one batch at a time."""
    from delta.tables import DeltaTable
    if not spark.catalog.tableExists(destination):
        updates.write.format("delta").mode("errorifexists").saveAsTable(destination)
        return
    condition = " AND ".join(f"saved.{validate_identifier(key)} = incoming.{key}" for key in keys)
    (
        DeltaTable.forName(spark, destination).alias("saved")
        .merge(updates.alias("incoming"), condition)
        .whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
    )


def write_monitoring_row(spark, destination: str, values: dict) -> None:
    """An explicit schema also supports failure rows without scored outputs."""
    from pyspark.sql.types import (
        StructType, StructField, StringType, TimestampType, LongType, DoubleType, BooleanType,
    )
    fields = {
        "batch_id": StringType(), "window_start": TimestampType(), "window_end": TimestampType(),
        "model_name": StringType(), "model_version": StringType(), "source_table_version": LongType(),
        "row_count": LongType(), "point_flag_rate": DoubleType(), "alert_count": LongType(),
        "max_mean_shift": DoubleType(), "data_lag_minutes": DoubleType(),
        "quality_status": StringType(), "drift_review": BooleanType(), "status": StringType(),
    }
    record = dict(values)
    for column in ["window_start", "window_end"]:
        record[column] = parse_time(record[column]).tz_localize(None).to_pydatetime()
    schema = StructType([StructField(name, kind, True) for name, kind in fields.items()])
    updates = spark.createDataFrame([tuple(record.get(name) for name in fields)], schema)
    merge_delta_table(spark, destination, updates, ["batch_id"])
