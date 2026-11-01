"""Score one UTC window with a pinned champion; upsert predictions and monitoring."""

import hashlib

import pandas as pd

from .databricks_helpers import (
    batch_window, job_parser, latest_table_version, merge_delta_table,
    spark_session, table_name, write_monitoring_row,
)
from .detection import FEATURE_COLUMNS, MODEL_INPUT_COLUMNS
from .dataset import CADENCE_MINUTES


def validate_spark_data(metric_data) -> None:
    from pyspark.sql import functions as F
    from pyspark.sql.types import NumericType, StringType, TimestampType
    column_types = {field.name: field.dataType for field in metric_data.schema.fields}
    required_columns = ["timestamp", "series_id", *MODEL_INPUT_COLUMNS]
    if set(required_columns) - set(column_types):
        raise ValueError("Source requires timestamp, series_id and a numeric value column")
    if not isinstance(column_types["timestamp"], TimestampType) or not isinstance(column_types["series_id"], StringType):
        raise ValueError("timestamp must be TIMESTAMP; series_id must be STRING")
    if not all(isinstance(column_types[name], NumericType) for name in MODEL_INPUT_COLUMNS):
        raise ValueError("Metric columns must have numeric types")
    invalid = F.col("timestamp").isNull() | F.col("series_id").isNull() | (F.trim("series_id") == "")
    for name in MODEL_INPUT_COLUMNS:
        invalid = invalid | F.col(name).isNull() | F.isnan(name) | (F.abs(F.col(name)) == float("inf"))
    if metric_data.where(invalid).limit(1).count():
        raise ValueError("Invalid metric rows; batch rejected before scoring")
    if metric_data.groupBy("series_id", "timestamp").count().where("count > 1").limit(1).count():
        raise ValueError("Duplicate series/timestamp keys; batch rejected")


def add_alert_columns(scored_data):
    """Spark equivalent of detection.apply_alert_policy, including gap resets."""
    from pyspark.sql import Window, functions as F
    chronological = Window.partitionBy("series_id").orderBy("timestamp")
    previous_time = F.lag("timestamp").over(chronological)
    interval_gap = previous_time.isNull() | (
        F.col("timestamp").cast("long") - previous_time.cast("long") != CADENCE_MINUTES * 60
    )
    alerts = scored_data.withColumn("interval_gap", interval_gap.cast("int"))
    alerts = alerts.withColumn("segment", F.sum("interval_gap").over(chronological))
    segment_order = Window.partitionBy("series_id", "segment").orderBy("timestamp")
    rolling_window = segment_order.rowsBetween(-2, 0)
    alerts = alerts.withColumn("alert_active", F.sum(F.col("point_flag").cast("int")).over(rolling_window) >= 2)
    alerts = alerts.withColumn("previous_active", F.lag("alert_active", 1, False).over(segment_order))
    return alerts.withColumn("alert_started", F.col("alert_active") & ~F.col("previous_active")).drop(
        "interval_gap", "segment", "previous_active",
    )


def main() -> None:
    parser = job_parser(__doc__)
    parser.add_argument("--model-alias", default="champion")
    parser.add_argument("--start-time", default="")
    parser.add_argument("--end-time", default="")
    arguments = parser.parse_args()
    if not arguments.schema.endswith("_dev") and arguments.model_alias != "champion":
        raise ValueError("Candidate inspection is development-only; production scores the champion")
    window_start, window_end = batch_window(arguments.start_time, arguments.end_time)
    import mlflow
    from mlflow import MlflowClient
    from pyspark.sql import functions as F

    source_table = table_name(arguments.catalog, arguments.schema, "metric_readings")
    prediction_table = table_name(arguments.catalog, arguments.schema, "metric_predictions")
    alert_table = table_name(arguments.catalog, arguments.schema, "metric_alerts")
    monitoring_table = table_name(arguments.catalog, arguments.schema, "batch_monitoring")
    model_name = table_name(arguments.catalog, arguments.schema, "metric_anomaly")
    spark = spark_session()
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    # Resolve once. An alias change during a batch cannot change its model.
    model_version = MlflowClient().get_model_version_by_alias(model_name, arguments.model_alias)
    if not arguments.schema.endswith("_dev"):
        if model_version.tags.get("validation_passed") != "true" or model_version.tags.get("data_source") != "reviewed_real":
            raise ValueError("Production scoring requires a validated model trained on reviewed real data")
    model_uri = f"models:/{model_name}/{model_version.version}"
    batch_id = hashlib.sha256(f"{source_table}|{window_start.isoformat()}|{window_end.isoformat()}|{model_uri}".encode()).hexdigest()[:24]
    monitoring = {
        "batch_id": batch_id, "window_start": window_start, "window_end": window_end,
        "model_name": model_name, "model_version": str(model_version.version),
        "quality_status": "not_checked", "status": "failed",
    }
    cached_predictions = None
    try:
        source_version = latest_table_version(spark, source_table)
        monitoring["source_table_version"] = source_version
        # Three preceding samples reconstruct episode starts at the batch boundary.
        context_start = window_start - pd.Timedelta(minutes=3 * CADENCE_MINUTES)
        metric_data = spark.read.option("versionAsOf", source_version).table(source_table).where(
            (F.col("timestamp") >= F.lit(context_start.to_pydatetime()))
            & (F.col("timestamp") < F.lit(window_end.to_pydatetime()))
        ).select("timestamp", "series_id", *MODEL_INPUT_COLUMNS)
        validate_spark_data(metric_data)
        monitoring["quality_status"] = "passed"
        score_model = mlflow.pyfunc.spark_udf(
            spark, model_uri=model_uri,
            result_type="struct<anomaly_score:double,is_anomaly:boolean>",
            env_manager="virtualenv",
        )
        scored = metric_data.withColumn("prediction", score_model(F.struct(*[F.col(name) for name in MODEL_INPUT_COLUMNS])))
        scored = scored.select("timestamp", "series_id", *MODEL_INPUT_COLUMNS,
                               F.col("prediction.anomaly_score").alias("anomaly_score"),
                               F.col("prediction.is_anomaly").alias("point_flag"))
        scored = add_alert_columns(scored).where(F.col("timestamp") >= F.lit(window_start.to_pydatetime()))
        scored = scored.withColumn("model_name", F.lit(model_name)).withColumn("model_version", F.lit(str(model_version.version)))
        scored = scored.withColumn("source_table_version", F.lit(source_version)).withColumn("batch_id", F.lit(batch_id))
        cached_predictions = scored.cache()
        summary = cached_predictions.agg(
            F.count("*").alias("row_count"), F.avg(F.col("point_flag").cast("double")).alias("point_flag_rate"),
            F.sum(F.col("alert_started").cast("long")).alias("alert_count"), F.max("timestamp").alias("latest_timestamp"),
        ).first()
        if not summary.row_count:
            raise ValueError("No readings in the requested batch window")
        # Feature monitoring uses the same fixed training reference as the model.
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        reference = loaded_model.unwrap_python_model().reference_statistics
        if not reference:
            raise ValueError("Model lacks training reference statistics; retrain using train_model")
        feature_data = cached_predictions
        feature_means = feature_data.agg(*[F.avg(name).alias(name) for name in FEATURE_COLUMNS]).first().asDict()
        mean_shifts = {name: abs(feature_means[name] - reference["mean"][name]) / reference["std"][name] for name in FEATURE_COLUMNS}
        monitoring.update({
            "row_count": int(summary.row_count), "point_flag_rate": float(summary.point_flag_rate),
            "alert_count": int(summary.alert_count), "max_mean_shift": float(max(mean_shifts.values())),
            "data_lag_minutes": max(0.0, (window_end - pd.Timestamp(summary.latest_timestamp, tz="UTC")).total_seconds() / 60),
            "drift_review": max(mean_shifts.values()) > 1.0, "status": "succeeded",
        })
        # Keys include model version so backfills never silently replace another model's outputs.
        output_keys = ["series_id", "timestamp", "model_name", "model_version"]
        merge_delta_table(spark, prediction_table, cached_predictions, output_keys)
        # Keep both active and inactive decisions. The SQL view filters starts, so
        # corrected reruns can clear a prior alert without leaving a stale record.
        alert_decisions = cached_predictions.select(*output_keys, "alert_active", "alert_started", "anomaly_score", "batch_id", "source_table_version")
        merge_delta_table(spark, alert_table, alert_decisions, output_keys)
        write_monitoring_row(spark, monitoring_table, monitoring)
        print(f"Scored {summary.row_count} rows with {model_uri}; batch_id={batch_id}")
    except Exception:
        monitoring["status"] = "failed"
        if monitoring["quality_status"] == "not_checked":
            monitoring["quality_status"] = "failed"
        # Retain the original exception if recording the failure also fails.
        try:
            write_monitoring_row(spark, monitoring_table, monitoring)
        except Exception as monitoring_error:
            print(f"Could not record failed batch: {monitoring_error}")
        raise
    finally:
        if cached_predictions is not None:
            cached_predictions.unpersist()


if __name__ == "__main__":
    main()
