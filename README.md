# MetroPT-3 anomaly detection

Three anomaly detectors for [MetroPT-3 compressor telemetry](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset):
raw Isolation Forest, Isolation Forest with rolling features, and a CNN autoencoder.
Models train on February's assumed-normal readings; maintenance reports provide
four failure windows for evaluation.

## Results

| Model | Validation events found | Test events found |
|---|---:|---:|
| Isolation Forest | 3/3 | 1/1 |
| Isolation Forest + rolling features | 2/3 | 0/1 |
| CNN autoencoder | 3/3 | 0/1 |

Validation selected the CNN for fewer unmatched alerts. Its threshold was too
high for July's failure, so release is blocked. Raw Isolation Forest detected
that event 138 minutes after onset. Models and thresholds were not changed using
test results.

[Full results](reports/model_comparison.md) · [Failure plot](reports/figures/failure_zoom.png)

## Run

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen
uv run metropt download
uv run metropt prepare
uv run pytest
uv run metropt run
```

Settings are in [configs/experiments.yaml](configs/experiments.yaml).

```text
data/                  downloaded and prepared sensor readings
artifacts/data/        data quality and EDA tables
artifacts/evaluation/  benchmark metrics
artifacts/runs/        local models and scored records (not committed)
mlruns/                MLflow database and model storage (not committed)
reports/               readable results and figures
```

Databricks jobs cover training, registration, batch scoring, monitoring, and
reviewed promotion. They have not run in a workspace. AutoML and Model Serving
are outside the implemented scope.

[Data notes](docs/data.md) · [Model choices](docs/decisions.md) ·
[Local operation](docs/operations.md) · [Databricks setup](databricks/README.md)
