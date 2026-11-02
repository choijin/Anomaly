# Data and evaluation

Source: [UCI MetroPT-3](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset),
[DOI: 10.24432/C5VW3R](https://doi.org/10.24432/C5VW3R), CC BY 4.0.
The download command verifies archive and CSV hashes pinned in
[`data.py`](../src/anomaly_detection/data.py).

## Observed data

| Check | Result |
|---|---:|
| Raw rows | 1,516,948 |
| One-minute snapshots | 252,719 |
| Median raw interval | 10 seconds |
| Gaps longer than 90 seconds | 331 |
| Longest gap | About 48 hours |
| Missing sensor values / duplicate timestamps | 0 / 0 |

There are seven continuous sensors and eight binary signals. `LPS`, `Oil_level`,
and `Caudal_impulses` are constant in training. Later changes request monitoring
review. Motor-current bands in the EDA describe operating states, not failures.
Full measurements are in [data_quality.json](../reports/data_quality.json).

Snapshots take the last complete reading available at each minute endpoint.
There is no interpolation or forward fill. Original gap boundaries survive in
`segment_id`, including gaps hidden between adjacent occupied minutes. A 60-step
window spans 59 minutes. Minute snapshots can discard shorter transients.

The source describes both 1 Hz logging and 0.1 Hz data; the CSV's median interval
supports 0.1 Hz. It extends to September 1 despite the February–August description.
No timezone is supplied, so timestamps retain the source clock.

## Failure reports

| Event | Start | Reported end | Partition |
|---|---|---|---|
| F1 | Apr 18 00:00 | Apr 18 23:59 | Validation |
| F2 | May 29 23:30 | May 30 06:00 | Validation |
| F3 | Jun 5 10:00 | Jun 7 14:30 | Validation |
| F4 | Jul 15 14:30 | Jul 15 19:00 | Test |

All dates are in 2020. Minute-resolution report ends include the whole final
minute: the implementation adds one minute and uses `[start, end)` intervals.
The source repeats report number #1 and puts a `30Apr` maintenance note beside
the May event. Project IDs distinguish the intervals; the inconsistent maintenance
note is not used as a label.

## Split and label policy

| Partition | Half-open interval | Rows before warm-up | Purpose |
|---|---|---:|---|
| Train | Feb 1–Feb 23 | 27,902 | Fit models and scalers |
| Normal development | Feb 23–Mar 1 | 7,625 | CNN early stopping |
| Validation | Mar 1–Jun 15 | 124,351 | Thresholds and model selection |
| Test | Jun 15–Sep 2 | 92,841 | Final assessment |

UCI recommends the first month for training; February is assumed normal under
that recommendation. Later observations are unknown unless covered by a failure
report. Configurable 24-hour guard bands exclude nearby failures from normal
training. Features exclude all annotations, timestamps, and identifiers.

Feature histories and sequence windows are built separately in each partition.
All models use the same 60-step eligible endpoints. Gaps and warm-up reduce scored
coverage; July's event has 211 scored minutes out of 271 reported minutes.

## Metrics and calibration

An event is detected when a persisted alert occurs inside its report interval.
Delay starts at reported onset and is defined only for detected events. Six-hour
pre-onset alerts are measured separately. No point adjustment is used.

Unmatched alerts/day divides alerts outside reports by observed, scored time
outside reports. Missing gaps add no exposure. Point precision, recall, PR-AUC,
and average precision use interval-derived proxy labels and are secondary metrics.
`pr_auc` uses trapezoidal integration; `average_precision` is reported separately.

Thresholds use a fixed validation quantile grid under a common one-alert/day
ceiling. Ranking prefers event recall, then fewer unmatched alerts, then a higher
threshold. Model ties favor simpler models. Selection is saved before test scoring.
Only three validation events and one test event are available from one asset.
