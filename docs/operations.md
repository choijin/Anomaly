# Operate the pipeline

## Release

1. Run software tests and build the wheel. Deploy development jobs.
2. Train on explicit chronological windows. Inspect MLflow evaluation and data lineage. The registered candidate carries its evaluation outcome; registration is not approval.
3. Score `model_alias=candidate` in development for inspection. The supplied benchmark models fail the example recall/false-alert gates and must not be promoted.
4. For a genuinely validated model, use the protected promotion workflow with its exact version and approval reference. The job records the previous champion and updates the alias last.
5. Score one production batch, inspect outputs, configure failure notifications, then enable the paused daily schedule.

Production requires passing validation and `data_source=reviewed_real`, substantiated by data review. The public benchmark cannot be relabeled into production evidence. Model/Job ownership and protected GitHub environments enforce who can approve releases.

## Recover a scoring batch

Start with the Databricks job exception and `batch_monitoring`. A missing/unapproved champion can fail before a monitoring row is created.

- Invalid input or duplicates: repair upstream data and rerun the same window.
- Partial output writes: rerun the window. Delta upserts prevent duplicate keys; consume only successful batches. Writes to separate tables are not one transaction.
- Late/corrected readings: replay the affected interval and the following 15 minutes to refresh alert boundary decisions. Source deletions need explicit output reconciliation; upserts do not delete missing rows.
- Drift or excess alerts: review input distributions and operational context. The mean-shift heuristic alone does not detect every kind of drift. No automatic retraining or champion changes occur.

`data_lag_minutes` measures freshness relative to a batch's window end. `databricks/monitoring.sql` separately checks current ingestion freshness; the historical benchmark is intentionally stale by today's clock.

## Roll back

Under a reviewed change, stop new scoring runs, inspect the champion's `previous_champion_version` tag and restore that explicit known-good version:

```python
import mlflow
from mlflow import MlflowClient
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
model_name = "YOUR_CATALOG.metric_anomaly_prod.metric_anomaly"
current = client.get_model_version_by_alias(model_name, "champion")
client.set_registered_model_alias(model_name, "champion", current.tags["previous_champion_version"])
```

In-flight batches remain pinned to their original model. Rerun affected windows as needed; downstream consumers must choose the approved model version. A Model Serving endpoint needs its own version update after an alias change.
