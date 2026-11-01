# Run on Databricks

The same Python package runs locally and as Databricks wheel jobs. The operational path is Delta input → MLflow/Unity Catalog candidate → reviewed champion → Delta predictions, alerts and monitoring.

No workspace was supplied. Configuration has been checked locally; Spark execution, registration and deployment still need a workspace smoke test.

## Development benchmark

Install uv and the [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install). Use an existing Unity Catalog catalog and a dedicated UC-enabled ML cluster (Runtime 15.4 LTS or newer with Python 3.11+ and PyPI access).

```bash
databricks auth login --host https://YOUR-WORKSPACE
export BUNDLE_VAR_catalog=main
export BUNDLE_VAR_existing_cluster_id=YOUR-CLUSTER-ID

databricks bundle validate --target dev
databricks bundle deploy --target dev
databricks bundle run --target dev load_data
databricks bundle run --target dev train_model --params \
  'train_start=2014-02-14T14:27:00Z,train_end=2014-02-17T14:27:00Z,calibration_end=2014-02-18T14:27:00Z,evaluation_end=2014-02-28T14:27:00Z'
databricks bundle run --target dev score_batch --params \
  'model_alias=candidate,start_time=2014-02-18T14:27:00Z,end_time=2014-02-28T14:27:00Z'
```

`load_data` creates the benchmark table once and refuses to overwrite it. The data ships inside the wheel, so no Kaggle credentials or separate file upload are needed.

Training uses explicit chronological windows, a pinned Delta snapshot and a 100,000-row driver limit. It logs inputs, evaluation, signature and reload parity, then registers a `candidate` with `validation_passed=true/false`. The example gates require at least 80% window recall and at most 0.1 false alerts per normal series-day.

**The supplied benchmark model fails these gates.** Development scoring uses `candidate` for inspection; training does not update `champion`. An attempted promotion is rejected. Do not lower acceptance limits or relabel benchmark data to portray it as a production-qualified model.

## Production setup

1. An administrator creates `metric_anomaly_prod` and grants a runtime service principal access. Set `BUNDLE_VAR_service_principal_id` to its application ID.
2. Supply independently reviewed history to a separate development `metric_readings` table and live input to production. Scoring needs TIMESTAMP `timestamp`, STRING `series_id`, and numeric `value`. Training also needs boolean `is_anomaly`, nonempty `event_id` on incident rows, and `data_source='reviewed_real'`. One model assumes comparable metric scales across its series.
3. Train and inspect a candidate on independent historical windows. Promote an explicit passing version with an approval reference. Benchmark provenance is development-only; `--allow-benchmark` cannot bypass validation checks.
4. Run a production window manually, inspect its outputs, configure failure notifications, then set `BUNDLE_VAR_scoring_schedule=UNPAUSED` and redeploy. The job scores the previous complete UTC day at 01:15 UTC. Historical benchmark dates require explicit backfill parameters.

```bash
export BUNDLE_VAR_service_principal_id=YOUR-SERVICE-PRINCIPAL-APPLICATION-ID
databricks bundle validate --target prod
databricks bundle deploy --target prod
databricks bundle run --target prod promote_model --params \
  "source_model=${BUNDLE_VAR_catalog}.metric_anomaly_dev.metric_anomaly,source_version=YOUR-VALIDATED-VERSION,approval_ticket=CHANGE-123"
databricks bundle run --target prod score_batch
```

Each batch pins a model version and a source Delta version. Three preceding readings preserve alert episodes across boundaries. Delta MERGE prevents duplicate retry keys. Invalid input and empty windows fail the scoring job; production also rejects unvalidated models and candidate aliases.

| Table | Purpose |
|---|---|
| `metric_readings` | Versioned source observations |
| `metric_predictions` | Scores, flags and model/data versions |
| `metric_alerts` | Decisions; filter `alert_started` for alert episode starts |
| `batch_monitoring` | Quality, row/alert counts, drift heuristic, data lag and batch status |

Run [monitoring.sql](monitoring.sql) with the environment's catalog/schema selected. Use [operations](../docs/operations.md) for failure recovery and rollback. Raw timestamps have no offset; the project stores them as UTC by convention, without asserting the original timezone.

The runtime identity needs cluster/experiment access, `USE CATALOG`, `USE SCHEMA`, input `SELECT`, and output `CREATE TABLE`/`MODIFY`. Training needs `CREATE MODEL`; alias management requires model ownership. Restrict production job execution and model ownership to approved operators.

## CI/CD

Use this directory as the repository root. CI installs the lockfile, runs tests and a small MLflow training/packaging check, and builds the wheel. This checks delivery mechanics; poor predictive performance does not fail software CI. Model promotion enforces the evaluation gate separately.

Create GitHub environments `dev` and `prod`, with required reviewers and restricted branches on `prod`. Set `DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_CATALOG`, and `DATABRICKS_CLUSTER_ID` as environment variables. Configure [GitHub OIDC federation](https://docs.databricks.com/aws/en/dev-tools/auth/provider-github). Workflow YAML does not create environment protections.

Manual deployment updates job definitions without starting training or promotion. The separate production promotion workflow chooses an explicit model version and approval reference. For failure emails, set `failure_emails` in `.databricks/bundle/<target>/variable-overrides.json`; configure the same override on deployment runners.

Optional notebooks: [train_automl.py](train_automl.py) requires a larger independently labeled history with both classes in each chronological split. This tiny benchmark is unsuitable for that exercise. [deploy_serving_endpoint.py](deploy_serving_endpoint.py) serves an approved champion for request-time scores; it is not required for daily batch scoring. Neither notebook has been executed here.

References: [wheel bundles](https://docs.databricks.com/aws/en/dev-tools/bundles/python-wheel/), [model lifecycle](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle).
