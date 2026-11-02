# MetroPT-3 anomaly detection

This project uses the [MetroPT-3 compressor dataset](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset)
to develop and compare three anomaly detectors: Isolation Forest, Isolation Forest
with rolling features, and a CNN autoencoder. I trained them on February's
assumed-normal readings and used four reported failures to evaluate them.

## Results

| Model | Validation events found | Test events found |
|---|---:|---:|
| Isolation Forest | 3/3 | 1/1 |
| Isolation Forest + rolling features | 2/3 | 0/1 |
| CNN autoencoder | 3/3 | 0/1 |

I selected the CNN for fewer unmatched validation alerts, but kept release blocked
after it missed July's failure. Although raw Isolation Forest caught that event
138 minutes after onset, I kept the original model choice and thresholds.

[Full results](reports/model_comparison.md) · [Failure plot](reports/figures/failure_zoom.png)

## Run

Run this with Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen
uv run metropt download
uv run metropt prepare
uv run pytest
uv run metropt run
```

Settings are in [configs/experiments.yaml](configs/experiments.yaml), with outputs organized here:

```text
data/                  downloaded and prepared sensor readings
artifacts/data/        data quality and EDA tables
artifacts/evaluation/  benchmark metrics
artifacts/runs/        local models and scored records (not committed)
mlruns/                MLflow database and model storage (not committed)
reports/               readable results and figures
```

I've configured Databricks training and batch deployment, but haven't run it
in a workspace yet. AutoML and Model Serving aren't included.

[Data notes](docs/data.md) · [Model choices](docs/decisions.md) ·
[Local operation](docs/operations.md) · [Databricks setup](databricks/README.md)
