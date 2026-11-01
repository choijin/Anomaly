"""Train from a pinned Delta snapshot; register an evaluated candidate for review."""

import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .databricks_helpers import (
    job_parser, latest_table_version, spark_session, table_name, validate_training_windows,
)
from .detection import (
    AnomalyDetector, MODEL_INPUT_COLUMNS, apply_alert_policy, build_features,
    evaluate_predictions, validate_dataset,
)


def passes_validation(metrics: dict, min_event_recall: float = 0.8, max_false_alerts_per_day: float = 0.1) -> bool:
    """Example acceptance contract; agree these limits with reliability engineers."""
    required = ["event_recall", "false_alerts_per_normal_day", "events_total", "normal_series_days"]
    if any(not isinstance(metrics.get(name), (int, float)) or not math.isfinite(metrics[name]) for name in required):
        return False
    return (
        metrics["events_total"] > 0 and metrics["normal_series_days"] > 0
        and 0 <= metrics["event_recall"] <= 1
        and metrics["event_recall"] >= min_event_recall
        and 0 <= metrics["false_alerts_per_normal_day"] <= max_false_alerts_per_day
    )


def validate_training_labels(metric_data: pd.DataFrame) -> str:
    for name in ["is_anomaly", "event_id", "data_source"]:
        if name not in metric_data or metric_data[name].isna().any():
            raise ValueError(f"Training/evaluation requires non-null {name}")
    if not pd.api.types.is_bool_dtype(metric_data.is_anomaly):
        raise ValueError("is_anomaly must be boolean")
    if metric_data.loc[metric_data.is_anomaly, "event_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Each labeled anomaly requires a nonempty event_id")
    sources = metric_data.data_source.unique().tolist()
    if len(sources) != 1 or sources[0] not in {"public_benchmark", "reviewed_real"}:
        raise ValueError("Use one data_source: public_benchmark or reviewed_real")
    return sources[0]


def main() -> None:
    parser = job_parser(__doc__)
    parser.add_argument("--experiment-path", required=True)
    for boundary in ["train-start", "train-end", "calibration-end", "evaluation-end"]:
        parser.add_argument(f"--{boundary}", required=True)
    parser.add_argument("--method", choices=["robust_zscore", "isolation_forest"], default="robust_zscore")
    parser.add_argument("--max-driver-rows", type=int, default=100000)
    arguments = parser.parse_args()
    boundaries = validate_training_windows(
        arguments.train_start, arguments.train_end, arguments.calibration_end, arguments.evaluation_end,
    )
    if arguments.max_driver_rows < 1:
        raise ValueError("max-driver-rows must be positive")
    import mlflow
    from mlflow import MlflowClient
    from mlflow.models import infer_signature
    from pyspark.sql import functions as F
    from .model_tracking import MetricModel

    source_table = table_name(arguments.catalog, arguments.schema, "metric_readings")
    model_name = table_name(arguments.catalog, arguments.schema, "metric_anomaly")
    spark = spark_session()
    source_version = latest_table_version(spark, source_table)
    source_data = spark.read.option("versionAsOf", source_version).table(source_table)
    source_data = source_data.where(
        (F.col("timestamp") >= F.lit(boundaries[0].to_pydatetime()))
        & (F.col("timestamp") < F.lit(boundaries[3].to_pydatetime()))
    )
    metric_data = source_data.orderBy("series_id", "timestamp").limit(arguments.max_driver_rows + 1).toPandas()
    if len(metric_data) > arguments.max_driver_rows:
        raise ValueError("Training collection exceeds max-driver-rows; narrow the dates or raise the reviewed limit")
    metric_data["timestamp"] = pd.to_datetime(metric_data.timestamp, utc=True)
    validate_dataset(metric_data)
    data_source = validate_training_labels(metric_data)
    training_data = metric_data.loc[metric_data.timestamp < boundaries[1]]
    calibration_data = metric_data.loc[(metric_data.timestamp >= boundaries[1]) & (metric_data.timestamp < boundaries[2])]
    evaluation_data = metric_data.loc[metric_data.timestamp >= boundaries[2]].copy()
    if any(part.empty for part in [training_data, calibration_data, evaluation_data]):
        raise ValueError("Each time window needs data")
    if len(training_data) < 2:
        raise ValueError("Training needs at least two observations to estimate reference variability")
    if training_data.is_anomaly.any() or calibration_data.is_anomaly.any():
        raise ValueError("Training and calibration windows must be background operation without labeled windows")
    # Incident IDs identify one series; healthy earlier windows prevent event leakage.
    for _, event in metric_data.loc[metric_data.is_anomaly].groupby("event_id"):
        if event.series_id.nunique() != 1:
            raise ValueError("event_id must uniquely identify an series incident")
    detector = AnomalyDetector(arguments.method).fit(training_data).calibrate(calibration_data)
    predictions = detector.predict(evaluation_data)
    evaluation_data["point_flag"] = predictions.is_anomaly
    evaluation_data["anomaly_score"] = predictions.anomaly_score
    metrics, event_details = evaluate_predictions(apply_alert_policy(evaluation_data))
    accepted = passes_validation(metrics)
    reference_features = build_features(training_data)
    reference_statistics = {
        "mean": reference_features.mean().to_dict(),
        "std": reference_features.std().clip(lower=1e-6).to_dict(),
    }
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(arguments.experiment_path)
    with mlflow.start_run(run_name=f"metric-{arguments.method}") as training_run:
        mlflow.log_params({
            "method": arguments.method, "seed": 42, "false_positive_budget": 0.005,
            "source_table": source_table, "source_table_version": source_version,
            "train_start": boundaries[0].isoformat(), "train_end": boundaries[1].isoformat(),
            "calibration_end": boundaries[2].isoformat(), "evaluation_end": boundaries[3].isoformat(),
            "min_event_recall": 0.8, "max_false_alerts_per_normal_day": 0.1,
        })
        mlflow.log_metrics({name: float(value) for name, value in metrics.items() if isinstance(value, (int, float)) and math.isfinite(value)})
        mlflow.set_tags({"data_source": data_source, "validation_passed": str(accepted).lower()})
        mlflow.log_dict({"events": event_details}, "evaluation_events.json")
        for context, window_start, window_end in zip(
            ["training", "calibration", "evaluation"], boundaries, boundaries[1:],
        ):
            split_data = source_data.where(
                (F.col("timestamp") >= F.lit(window_start.to_pydatetime()))
                & (F.col("timestamp") < F.lit(window_end.to_pydatetime()))
            )
            dataset = mlflow.data.from_spark(
                split_data, table_name=source_table, version=str(source_version), name=context,
            )
            mlflow.log_input(dataset, context=context)
        input_example = evaluation_data[MODEL_INPUT_COLUMNS].head(5)
        model_info = mlflow.pyfunc.log_model(
            name="detector", python_model=MetricModel(detector, reference_statistics),
            signature=infer_signature(input_example, detector.predict(input_example)),
            input_example=input_example, code_paths=[str(Path(__file__).parent)],
            pip_requirements=[f"{name}=={importlib.metadata.version(name)}" for name in ["mlflow", "numpy", "pandas", "scikit-learn", "joblib"]],
        )
        loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
        restored_predictions = loaded_model.predict(evaluation_data[MODEL_INPUT_COLUMNS])
        np.testing.assert_allclose(restored_predictions.anomaly_score, predictions.anomaly_score)
        np.testing.assert_array_equal(restored_predictions.is_anomaly, predictions.is_anomaly)
        candidate = mlflow.register_model(model_info.model_uri, model_name)
        client = MlflowClient()
        for name, value in {
            "validation_passed": str(accepted).lower(), "data_source": data_source,
            "source_table": source_table, "source_table_version": str(source_version),
            "training_run_id": training_run.info.run_id,
        }.items():
            client.set_model_version_tag(model_name, candidate.version, name, value)
        client.set_registered_model_alias(model_name, "candidate", candidate.version)
        print(json.dumps({"model_name": model_name, "candidate_version": candidate.version,
                          "validation_passed": accepted, "run_id": training_run.info.run_id, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
