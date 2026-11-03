# Compressor anomaly detection

This project uses [MetroPT-3 sensor readings](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset)
to compare three ways of flagging unusual compressor behavior for maintenance
review. I also built a repeatable workflow for tracking experiments, packaging
models, scoring new readings, and checking whether a model is suitable to release.

## What I found

I trained the detectors on an assumed healthy period, used three reported failures
to choose the model and its alert cutoff, and held back a fourth for a final check.

| Approach | What it uses | Development: failures found | Final check |
|---|---|---:|---:|
| Isolation Forest | Individual sensor readings | 3/3 | 1/1 |
| Isolation Forest + history | Readings and recent trends | 2/3 | 0/1 |
| CNN autoencoder | Sequences of readings | 3/3 | 0/1 |

The CNN produced fewer alerts outside known failures during development, but
missed the final failure because its alert cutoff was too high. The basic
Isolation Forest found that failure 138 minutes after it started.

That missed failure stops the CNN from passing the release checks. I'd evaluate
a revised model or cutoff on fresh data before using it for maintenance alerts.

[Detailed results](reports/model_comparison.md) · [Failure plot](reports/figures/failure_zoom.png)

## Reproduce the study

Use Python 3.12 and [uv](https://docs.astral.sh/uv/) from this checkout:

```bash
uv sync --frozen
uv run python -m research download
uv run python -m research prepare
uv run pytest
uv run python -m research run
```

[Configuration](configs/experiments.yaml) fixes the data splits, seeds, and alert
rules. Research tools stay separate from the installed runtime:

```text
research/                  EDA, notebooks, comparisons, and plots
src/anomaly_detection/
  data/                    ingestion, validation, and causal features
  models/                  detector implementations
  training/                fitting, calibration, and evaluation
  inference/               scoring, alert rules, and monitoring
  mlops/                   MLflow, Databricks jobs, and promotion checks
```

Every push tests the data handling, alert logic, and model approval rules, then
builds the runtime wheel with research code excluded. Databricks jobs use that
same package; I haven't executed them in a workspace.

[Data notes](docs/data.md) · [Model choices](docs/decisions.md) ·
[Running the model](docs/operations.md) · [Databricks setup](databricks/README.md)
