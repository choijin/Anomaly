# Local operation

`uv run metropt run` writes models, thresholds, scores, and configuration to
`artifacts/runs/benchmark/`. Summary metrics go to `artifacts/evaluation/`;
readable output goes to `reports/`.

MLflow stores experiment records in `mlruns/mlflow.db` and packaged models under
`mlruns/`. Open the local UI with:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5001
```

To score prepared data:

```bash
uv run metropt predict \
  --model artifacts/runs/benchmark/selected_model.joblib \
  --input data/processed/observations.parquet \
  --output artifacts/runs/replay_predictions.parquet
```

Supply ordered minute snapshots with `timestamp`, `segment_id`, and all 15 sensors.
For exact test reproduction, use the same test-partition start. Arbitrarily
trimming history changes window warm-up and can change alert cooldown.

## Databricks batches and releases

Jobs pin the Delta snapshot and resolve the model alias once. They replay
`[replay_start, end)` and write `[start, end)`. Keep the replay anchor fixed.
Retry writes merge on batch ID and timestamp. The driver limit is 500,000 rows;
larger deployments need persisted per-asset state.

CI tests and builds the wheel. Deployment and model promotion are separate.
Promotion needs a version, approval reference, passing validation, and
`release_blocked=false`. Production also requires site-reviewed data.
The current CNN cannot become champion. Rollback promotes an approved older
version recorded in `previous_champion_version`.

Monitor input quality, gaps, unscorable rows, drift, latency, and alert counts
with [monitoring.sql](../databricks/monitoring.sql). Investigate ingestion and
operating states before changing thresholds. Maintenance review is needed to
measure accuracy. See [Databricks setup](../databricks/README.md) for deployment.
