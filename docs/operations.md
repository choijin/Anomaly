# Running the model

The research commands reproduce the study from the checkout; they aren't part
of the deployed package. `python -m research run` calls the shared training
pipeline, then adds diagnostics, figures, and reports. Databricks training calls
that pipeline directly and records metrics and models without running EDA.

```text
data/                  downloaded and prepared readings
artifacts/data/        numeric data checks
artifacts/evaluation/  published benchmark metrics
artifacts/runs/        local models and scored records (ignored by Git)
mlruns/                tracking database and model storage (ignored by Git)
reports/               readable results and figures
```

To inspect tracked experiments:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5001
```

To score prepared readings using the installed runtime:

```bash
uv run metropt score \
  --model artifacts/runs/benchmark/selected_model.joblib \
  --input data/processed/observations.parquet \
  --output artifacts/runs/replay_predictions.parquet
```

Input needs ordered minute snapshots with `timestamp`, `segment_id`, and all 15
sensors. Keep the same replay start when comparing decisions; trimming history
changes warm-up and can change alert timing.

Databricks batches pin a Delta snapshot and resolve the model version once.
They read `[replay_start, end)` and write `[start, end)`, merging retry results
by batch ID and timestamp. The driver limit is 500,000 rows; scaling beyond that
would need persisted state per asset.

Deployment and model approval are separate. Approval requires an explicit version,
review reference, passing validation, and `release_blocked=false`. Production
also needs site-reviewed data. The current CNN fails that approval check.
Rollback uses an approved older version recorded in `previous_champion_version`.

[Monitoring SQL](../databricks/monitoring.sql) checks data quality, gaps, unscorable
rows, drift, latency, and alert counts. I'd investigate ingestion and operating
states before adjusting thresholds; accuracy needs maintenance review.
