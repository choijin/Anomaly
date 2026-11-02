# Operations

## Build and release

CI installs `uv.lock`, verifies the exported Databricks requirements, runs tests,
and builds a wheel. The manual benchmark workflow downloads the data and runs
all three experiments. Deployment and model promotion are separate workflows.

Training runs in development against a pinned Delta snapshot. It logs MLflow
runs, freezes validation selection, evaluates test, and registers the selected
artifact as `candidate`. Promotion requires:

- An explicit version and approval reference.
- Passing validation metrics and `release_blocked=false`.
- A matching configuration digest and retained source lineage.
- Site-reviewed provenance for production; public benchmarks are development-only.

The current CNN candidate missed July's event and cannot become `champion`.
A missing release-review outcome also blocks promotion. The alias changes last,
after all checks and audit tags succeed.

GitHub deployment uses `dev`/`prod` environments and OIDC. Configure
`DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_CATALOG`,
`DATABRICKS_CLUSTER_ID`, and `METROPT_RAW_DATA_PATH`; require reviewers for `prod`.
See [Databricks setup](../databricks/README.md).

## Inference contract

Input is one asset's ordered minute snapshots: `timestamp`, integer `segment_id`,
and the 15 sensors. Timestamps use the source clock; digital values must be 0/1.
Gaps reset feature windows and alert persistence. Cooldown retains the last
notification time. Warm-up rows return no score and `is_scorable=false`.

A request reads `[replay_start, end)` and emits `[start, end)`. Keep the same
replay anchor across batches; an arbitrary hour of history can change cooldown
phase. Delta and model versions are fixed for the request. Retry writes merge
on batch ID and timestamp. Overlapping requests retain their own batch IDs.

Scoring runs on one driver with a 500,000-row bound. Larger deployments should
persist rolling, sequence, and alert state per asset. Scheduling requires ingestion
watermarks and an explicit source version; this historical benchmark has no schedule.

## Monitoring and incidents

Successful batches write versioned predictions and monitoring JSON to Delta.
Check schema failures, nonfinite inputs, gaps, unscorable coverage, mean shifts,
changes in previously constant sensors, latency, and alert volume. Invalid input
fails the job; logs and configured failure emails provide the failure record.
[monitoring.sql](../databricks/monitoring.sql) queries the batch metrics.

For an alert increase, check ingestion and gaps first, then operating states and
drift against the training reference. Request maintenance review before changing
thresholds or retraining. Drift and alert counts do not establish model accuracy.

To roll back, promote the recorded `previous_champion_version` with an explicit
rollback approval reference. Record the incident and replay the affected batch
against its original Delta version using the restored model.
