# Databricks notebook source
# Optional exercise on Runtime 17.3 LTS ML with the project wheel installed.
# Requires independently reviewed real incidents in every chronological window.
# Benchmark window labels are unsuitable for supervised model selection.
# https://docs.databricks.com/aws/en/machine-learning/automl/automl-api-reference
import databricks.automl
import pandas as pd
from pyspark.sql import functions as F

from metric_anomaly.detection import MODEL_INPUT_COLUMNS

dbutils.widgets.text("source_table", "")
dbutils.widgets.text("training_end", "")  # Exclusive UTC timestamp; reserve later data.
source_table = dbutils.widgets.get("source_table").strip()
training_end = pd.Timestamp(dbutils.widgets.get("training_end"))
if not source_table or pd.isna(training_end) or training_end.tzinfo is None:
    raise ValueError("Supply a reviewed source_table and timezone-aware training_end")

reviewed_data = spark.table(source_table)
invalid_source = F.col("data_source").isNull() | (F.col("data_source") != "reviewed_real")
if reviewed_data.filter(invalid_source).limit(1).count():
    raise ValueError("All rows must have data_source='reviewed_real'")
training_data = (
    reviewed_data.filter(F.col("timestamp") < F.lit(training_end.to_pydatetime()))
    .select(*MODEL_INPUT_COLUMNS, "timestamp", "is_anomaly")
    .limit(100001).toPandas().sort_values("timestamp")
)
if len(training_data) > 100000:
    raise ValueError("Use a reviewed history window of at most 100,000 rows for this small exercise")
if training_data.empty or training_data.is_anomaly.isna().any():
    raise ValueError("Training history must be nonempty with no null labels")
if set(training_data.is_anomaly.unique()) != {False, True}:
    raise ValueError("Reviewed is_anomaly labels must contain both boolean classes")

# Preflight chronological 60/20/20 windows; inspect actual generated splits too.
validation_start, test_start = training_data.timestamp.quantile([0.6, 0.8])
windows = [
    training_data.loc[training_data.timestamp < validation_start],
    training_data.loc[training_data.timestamp.between(validation_start, test_start, inclusive="left")],
    training_data.loc[training_data.timestamp >= test_start],
]
for window_name, window_data in zip(["train", "validation", "test"], windows):
    if set(window_data.is_anomaly.unique()) != {False, True}:
        raise ValueError(f"{window_name} needs reviewed normal and incident examples")

# Later rows stay outside AutoML as an untouched acceptance holdout.
summary = databricks.automl.classify(
    training_data, target_col="is_anomaly", time_col="timestamp",
    primary_metric="f1", timeout_minutes=15,
)
# Inspect trial notebooks, class coverage, incident recall, and false alerts.
# No automatic registration or promotion; evaluate the external holdout once.
print(summary.best_trial.notebook_path)
