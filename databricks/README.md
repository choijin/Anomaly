# Databricks setup

I've configured this workflow but haven't run it in a workspace yet. It needs
Unity Catalog, a Volume, and Python 3.12 ML compute, such as Runtime 16.4 LTS ML.
The job identity needs table/model permissions and Volume access.

## Deploy

```bash
uv sync --frozen
uv run python -m research download
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

The load job verifies the CSV before creating `sensor_observations` and stops
if the table exists. Use its returned Delta version if it differs from `0`.
Training registers `catalog.metropt_dev.compressor_anomaly` as `candidate`;
scoring writes `anomaly_predictions` and `anomaly_monitoring`.

The selected CNN is available for development inspection, but its failed final
check prevents approval for operational use. For a future passing version:

```bash
databricks bundle run --target dev promote_model \
  --params 'source_model=YOUR_CATALOG.metropt_dev.compressor_anomaly,source_version=VERSION,approval_reference=CHANGE_ID,allow_benchmark=true'
```

Production needs a separate schema/service principal and site-reviewed data.
For GitHub deployment, use OIDC and `dev`/`prod` environments with required `prod`
reviewers. Set `DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_CATALOG`,
`DATABRICKS_CLUSTER_ID`, and `METROPT_RAW_DATA_PATH` in those environments.

## Dependencies

The wheel contains runtime code only; research scripts and notebooks are excluded
from deployment. Each task installs it alongside `requirements/databricks.txt`.
When updating `uv.lock`, regenerate the requirements too; CI checks that they match:

```bash
uv export --frozen --no-dev --no-emit-project --no-hashes \
  --emit-index-url --no-annotate -o requirements/databricks.txt
```

[Batch and release details](../docs/operations.md) ·
[Databricks library support](https://docs.databricks.com/aws/en/libraries/)
