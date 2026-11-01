# Server-metric anomaly detection and MLOps

A small project using **real AWS CloudWatch CPU data from the Numenta Anomaly Benchmark (NAB)**. It compares a robust z-score baseline with Isolation Forest, then packages the same scoring code for Databricks.

**Load data → train and evaluate → register candidate → approve promotion → batch score → monitor.**

## Run locally

From this directory, with Python 3.11–3.13 and uv:

```bash
uv sync --python 3.13 --extra tracking --frozen
uv run --extra tracking python -m metric_anomaly --track
uv run --extra tracking python -m unittest discover -s tests -v
uv build --wheel
```

The dataset and labels are bundled with the package; running the demo does not download data or require a Kaggle account. Open `artifacts/report.md` and `artifacts/report.png`.

```bash
uv run --extra tracking mlflow ui \
  --backend-store-uri sqlite:///artifacts/mlflow.db --host 127.0.0.1 --port 5001
```

## Dataset and model

- Source: [NAB AWS CloudWatch data](https://github.com/numenta/NAB/tree/master/data/realAWSCloudwatch), also in the [Kaggle NAB collection](https://www.kaggle.com/datasets/boltzmannbrain/nab).
- File: `ec2_cpu_utilization_5f5533.csv`; 4,032 five-minute readings and two published anomaly windows.
- Input contract: `timestamp`, `series_id`, `value`. The model uses only `value`; labels remain evaluation metadata.
- Split: first three days train, next day calibrate, remaining ten days evaluate. Training/calibration overlap no published anomaly windows. Absence of a window is not verified normal operation.
- Threshold: each model's 99.5th-percentile background calibration score. Alerts require two flags among three consecutive readings per series; gaps reset the rule.
- Metrics: approximate point precision/recall, window recall, detected-window delay, and false alert episodes per normal series-day. These are custom metrics, not official NAB scoring.

**Current result:** the baseline detects 1/2 windows with about 9.18 false alert episodes per normal series-day. Isolation Forest starts no persistent alert episodes. Both fail the example release gates. The report preserves these results; no threshold or model was retuned on the evaluation period.

This simple model detects unusual values, not all changes in temporal behavior. The project demonstrates a working delivery pipeline and a real model rejection. Two broad annotation windows do not establish production reliability or failure prediction.

## Files

Python modules are in `src/metric_anomaly/`:

| File | Responsibility |
|---|---|
| `dataset.py`, `datasets/` | Real benchmark, provenance, checksum, labels and chronological split |
| `detection.py` | Input checks, detectors, alert policy and evaluation |
| `run_demo.py` | Local comparison, model export and results plot |
| `model_tracking.py` | MLflow packaging and prediction reload check |
| `load_data.py` | Load the bundled benchmark into development Delta once |
| `train_model.py` | Pin a Delta snapshot, evaluate and register a tagged candidate |
| `promote_model.py` | Require passing validation and explicit approval; retain rollback metadata |
| `score_batch.py` | Spark scoring, idempotent Delta outputs and monitoring |
| `databricks_helpers.py` | Shared date, identifier and Delta-write utilities |
| `databricks.yml`, `resources/` | Development/production job configurations |
| `.github/workflows/` | CI checks, wheel build, manual deployment and model promotion |

## Databricks and interview use

Start with [Databricks setup](databricks/README.md), [interview guide](docs/interview.md), and [operations](docs/operations.md). No workspace was provided, so cloud execution and GitHub workflows remain untested.

Training registers candidates for inspection and tags their evaluation outcome. Development may batch-score a rejected candidate; promotion checks block it from becoming champion. Public benchmark provenance is also restricted to development. Production requires independently reviewed operating data and an approved validated model. Scoring schedules start paused.

| Role requirement | Project evidence |
|---|---|
| Reusable ML delivery | Shared Python wheel and Databricks jobs |
| MLflow / Unity Catalog | Tracking, model signature, candidate/champion versions and approval checks |
| CI/CD | Code tests and packaging; separate manual job deployment and model promotion |
| Data quality / lineage | Schema checks, raw-file checksum, Delta version pinning and model/data IDs |
| Monitoring / reliability | Quality status, flag volume, mean-shift review, freshness and rollback playbook |
| SQL / Spark | Distributed scoring, Delta upserts and monitoring queries |
| AutoML / Model Serving | Optional notebooks for later workspace exercises; not executed here |

CI verifies that the code can train, package and score. It can pass while model evaluation fails. Only a separately validated and approved model may become the production champion.
