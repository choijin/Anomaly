# Local operation

Running `uv run metropt run` saves models, thresholds, scores, and configuration
in `artifacts/runs/benchmark/`, with summary metrics in `artifacts/evaluation/`
and readable results in `reports/`.

MLflow keeps experiment records in `mlruns/mlflow.db`, which you can browse with:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5001
```

To score prepared data, run:

```bash
uv run metropt predict \
  --model artifacts/runs/benchmark/selected_model.joblib \
  --input data/processed/observations.parquet \
  --output artifacts/runs/replay_predictions.parquet
```

Pass ordered minute snapshots containing `timestamp`, `segment_id`, and all 15
sensors. To reproduce test decisions, start at the same test-partition boundary;
trimming history changes warm-up and can change alert cooldown.

## Databricks batches and releases

Each job pins a Delta snapshot and resolves the model alias once. It reads
`[replay_start, end)` but writes only `[start, end)`. Keeping the replay anchor
fixed preserves history, while merging on batch ID and timestamp handles retries.
The driver limit is 500,000 rows; beyond that, I'd persist state per asset.

CI tests and builds the wheel, while deployment and model promotion remain
separate. Promotion requires a version, approval reference, passing validation,
and `release_blocked=false`; production also needs site-reviewed data. The current
CNN can't become champion. To roll back, promote the approved older version in
`previous_champion_version`.

I monitor input quality, gaps, unscorable rows, drift, latency, and alert counts
through [monitoring.sql](../databricks/monitoring.sql). I'd check ingestion and
operating states before changing thresholds, and use maintenance review to assess
accuracy.
