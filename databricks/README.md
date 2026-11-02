# Databricks setup

The bundle uses the same Python package as the local run. It has been checked
locally but has not run in a Databricks workspace.

## Configure and deploy

Use an existing Unity Catalog catalog, `metropt_dev` schema, Volume, and compatible
ML compute with Python 3.12 (for example Runtime 16.4 LTS ML). The job identity
needs table/model permissions and Volume read access. Production uses a separate
schema and service principal.

```bash
uv sync --frozen
uv run metropt download
export DATABRICKS_HOST='https://YOUR_WORKSPACE'
databricks auth login --host "$DATABRICKS_HOST"
export BUNDLE_VAR_catalog='YOUR_CATALOG'
export BUNDLE_VAR_existing_cluster_id='YOUR_CLUSTER_ID'
export BUNDLE_VAR_raw_data_path='/Volumes/YOUR_CATALOG/metropt_dev/YOUR_VOLUME/MetroPT3(AirCompressor).csv'
databricks fs cp 'data/raw/MetroPT3(AirCompressor).csv' "dbfs:$BUNDLE_VAR_raw_data_path"
databricks bundle validate --target dev
databricks bundle deploy --target dev
```

`dbfs:/Volumes/...` addresses the Unity Catalog Volume. Code is deployed as a
wheel; raw data and fitted artifacts are excluded from bundle sync. Each task
installs the lockfile-exported runtime requirements alongside the wheel.

## Train and inspect

```bash
databricks bundle run --target dev load_data
databricks bundle run --target dev train_model
databricks bundle run --target dev score_batch \
  --params 'start=2020-06-15 00:00:00,end=2020-09-02 00:00:00,replay_start=2020-06-15 00:00:00,source_version=0,model_alias=candidate'
```

Loading verifies the CSV hash, prepares minute snapshots, and creates
`sensor_observations`. It fails if the table already exists. Use the actual Delta
version returned by loading in place of `source_version=0` if it differs.

Training selects on validation before reporting test results, logs Delta lineage,
and registers `catalog.metropt_dev.compressor_anomaly` as `candidate`. The current
CNN is release-blocked; candidate scoring is available for development inspection.
Outputs are `anomaly_predictions` and `anomaly_monitoring`.

Batch requests pin the Delta version and resolve the model alias once. They
replay a fixed session prefix, then keep only the requested half-open range.
The driver limit is 500,000 rows. Preserve `segment_id` and the replay anchor;
see the [inference contract](../docs/operations.md).

## Promote and roll back

For a future version that passes validation and release review:

```bash
databricks bundle run --target dev promote_model \
  --params 'source_model=YOUR_CATALOG.metropt_dev.compressor_anomaly,source_version=VERSION,approval_reference=CHANGE_ID,allow_benchmark=true'
```

The current candidate cannot pass this check. Production additionally requires
site-reviewed data; MetroPT benchmark provenance is restricted to development.
Promotion records the previous champion and changes the alias last. Rollback
uses the same job with an approved older version.

## Dependencies and scope

After changing dependencies, update `uv.lock` and regenerate:

```bash
uv export --frozen --no-dev --no-emit-project --no-hashes \
  --emit-index-url --no-annotate -o requirements/databricks.txt
```

Commit both files together. CI checks the export for drift. Requirements files
need Runtime 15.0 or later; verify package compatibility in the workspace.

Batch scoring is implemented. Model Serving would need ordered context or a
state service. AutoML does not directly provide these Isolation Forest/CNN
experiments and is not included.

References: [wheel tasks](https://docs.databricks.com/aws/en/dev-tools/bundles/job-task-types),
[Unity Catalog models](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/),
[library installation](https://docs.databricks.com/aws/en/libraries/).
