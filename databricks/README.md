# Databricks setup

This workflow is configured but has not run in a workspace. It needs Unity
Catalog, a Volume, and compatible Python 3.12 ML compute (for example Runtime
16.4 LTS ML). The job identity needs table/model permissions and Volume access.

## Deploy

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

## Run

```bash
databricks bundle run --target dev load_data
databricks bundle run --target dev train_model
databricks bundle run --target dev score_batch \
  --params 'start=2020-06-15 00:00:00,end=2020-09-02 00:00:00,replay_start=2020-06-15 00:00:00,source_version=0,model_alias=candidate'
```

Loading verifies the CSV and creates `sensor_observations`; it fails if the table
already exists. Replace `source_version=0` with the returned Delta version if needed.
Training registers `catalog.metropt_dev.compressor_anomaly` as `candidate`.
Scoring writes `anomaly_predictions` and `anomaly_monitoring`.

The CNN is blocked from release but can be inspected through `candidate`.
A future passing version can be promoted with:

```bash
databricks bundle run --target dev promote_model \
  --params 'source_model=YOUR_CATALOG.metropt_dev.compressor_anomaly,source_version=VERSION,approval_reference=CHANGE_ID,allow_benchmark=true'
```

Production uses a separate schema/service principal and requires site-reviewed
data. GitHub deployment uses OIDC and `dev`/`prod` environments; require reviewers
for `prod`. Set `DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_CATALOG`,
`DATABRICKS_CLUSTER_ID`, and `METROPT_RAW_DATA_PATH` in those environments.

## Dependencies

Tasks install the wheel and `requirements/databricks.txt`. After updating `uv.lock`,
regenerate that file; CI checks it for drift:

```bash
uv export --frozen --no-dev --no-emit-project --no-hashes \
  --emit-index-url --no-annotate -o requirements/databricks.txt
```

[Batch and release details](../docs/operations.md) ·
[Databricks library support](https://docs.databricks.com/aws/en/libraries/)
