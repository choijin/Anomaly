# Interview walkthrough

## A short pitch

> I built an anomaly-detection delivery pipeline around public AWS CloudWatch CPU data from NAB. I compared Isolation Forest with a robust z-score baseline using chronological training, calibration and evaluation windows. I packaged inference with MLflow and supplied Databricks jobs for training, governed promotion, scoring and monitoring.
>
> The real-data results were weak: the baseline caught one of two benchmark windows with excessive false alerts. I kept that result and made sure the model could not pass the release gate. Passing software tests and producing a model artifact do not establish that a model should serve production traffic.
>
> Local execution, serialization and release guards are tested. Databricks deployment, AutoML and Model Serving still need workspace execution. I would not claim production impact or hands-on cloud execution from this project alone.

## Five-minute demo

1. Run `python -m metric_anomaly --track` after installing the tracking extra.
2. Open `artifacts/report.png`: raw CPU values, background/evaluation boundaries, published windows and alert starts.
3. Show `metrics.json` and the model rejection. Explain that NAB windows are broad annotations; point metrics are approximate and these metrics are not official NAB scoring.
4. Show the MLflow run: input snapshots, parameters, metrics, signature, model artifact and reload parity.
5. Walk through `train_model.py`, `promote_model.py`, and `score_batch.py`. Explain version pinning, Delta upserts, mean-shift/volume monitoring and rollback.

## Questions to prepare for

**Why these algorithms?** A robust z-score measures distance from a training median in robust scale units. Isolation Forest isolates unusual observations using random partitions. Both are simple stateless value detectors; they can miss anomalies involving temporal patterns or variance rather than individual levels.

**How did you prevent leakage?** Time splits precede fitting. Training learns the model/statistics; later background calibration determines the threshold; the final period evaluates it. Labels never become features. Changing the design after examining results requires another untouched final dataset.

**How were alerts evaluated?** Two flags among three consecutive readings start an alert episode, separately per series. Gaps reset history. A benchmark window is detected only if an episode starts inside it. Delay is measured from the annotated window start, not before a failure. False alerts count episode starts outside windows per observed normal series-day.

**Why does CI pass when the model is rejected?** CI tests code and packaging contracts. The model evaluation checks whether predictive behavior is acceptable. A separate approval step checks evaluation and provenance before changing the champion. This separates a valid software release from a valid model release.

**How does Databricks fit?** A wheel bundle supplies development and production jobs. Training pins Delta data and records MLflow lineage. Unity Catalog holds evaluated candidates; approval controls the champion alias. Batch scoring resolves one immutable model version and writes versioned predictions, alert decisions and monitoring to Delta. These paths await cloud execution.

**What is the main limitation?** One short series and two broad windows cannot establish general reliability. The current one-value models perform poorly, and the short calibration period does not maintain its target flag rate on later data. More independent data and temporal features are possible follow-up work, not demonstrated improvements.

**What about AutoML and serving?** Optional notebooks are included. AutoML is a supervised challenger once enough trustworthy labels exist; this benchmark is too small for sound chronological classification. A serving endpoint is useful for request-time scores, while the current use case is periodic batch scoring.

**How would you operate it?** Monitor input quality, freshness, flag/alert volume and distribution changes. Investigate before retraining. Keep data/model versions on outputs, require reviewed promotion, and restore a previous approved champion during an incident. See [operations](operations.md).
