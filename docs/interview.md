# Project walkthrough

## Overview

I compared two Isolation Forest configurations and a small CNN autoencoder on
MetroPT-3 compressor telemetry. The models learn from an assumed-normal period;
maintenance reports support event-based evaluation. I used chronological splits,
train-only preprocessing, causal features, and explicit handling of data gaps.

I tracked each experiment in MLflow and packaged the model with its features,
scaler, threshold, and alert policy. The Databricks batch workflow includes Delta
versions, Unity Catalog registration, controlled promotion, and monitoring.
Local execution is verified; Databricks deployment still needs a workspace.

## Why did the CNN miss the test event?

It detected 3/3 validation events with 0.628 unmatched alerts/day, compared with
0.961 for raw Isolation Forest. That determined the model selection.

The CNN threshold was 32.52. Its July-event scores reached 8.83, and the maximum
across the entire test period was 11.29. Consequently it emitted no test alerts.
Its test PR-AUC was nevertheless 0.380 versus 0.285 for raw Isolation Forest:
ranking readings and choosing a working alert threshold are different problems.

The three validation events did not provide a representative calibration for
July. Operating-state changes and limited training variety may contribute, but
this experiment does not isolate their causes. The result does not prove that
CNNs are generally inferior. It shows that this model and threshold failed to
transfer to the held-out event. I kept the result and blocked promotion rather
than selecting another model using test performance.

## Modeling choices

- **Isolation Forest:** randomized partitions isolate unusual readings in fewer
  splits. Negated `score_samples` makes larger scores more anomalous.
- **Temporal features:** add rolling levels, variability, deviations, extremes,
  differences, and slopes over 1, 5, and 15 minutes. They did not improve this run.
- **CNN:** reconstructs multivariate sequences using continuous-channel MSE and
  binary-channel cross entropy. Scaling fits training only; early stopping uses
  a separate normal period. More complexity does not guarantee better detection.

## Engineering decisions

- Package preprocessing and alert rules with weights to keep training and batch
  inference consistent. Reload checks compare scores and alerts.
- Pin the Delta snapshot and resolve the model alias once per batch. Replay from
  a fixed session start so rolling history and cooldown remain consistent.
- Keep software deployment separate from model promotion. A passing CI job does
  not mean the model meets detection requirements.
- Monitor schema, missing data, gaps, warm-up coverage, drift, latency, and alert
  counts. Measuring accuracy requires reviewed maintenance events.
- For many assets, replace full-history replay with persisted per-asset state.
  A serving endpoint would need ordered context or a state service.

## Limits

One held-out event cannot establish general performance. Unreported periods
are unknown, and unmatched alerts are not confirmed false positives. Raw
Isolation Forest detected July's event after 138 minutes; that is detection,
not early warning. AutoML is outside this project's implemented scope.
