# Data

[MetroPT-3](https://doi.org/10.24432/C5VW3R) has 1,516,948 readings from seven
continuous and eight binary sensors. The CSV has no null readings, but it has
331 gaps longer than 90 seconds. The median sampling interval is 10 seconds.

The pipeline takes the last available reading each minute without filling gaps.
`segment_id` preserves breaks, even when adjacent minute bins hide a native gap.
All models use the same complete 60-step windows, built separately in each split.
A window spans 59 minutes. Scalers fit training data only; identifiers and report
annotations never enter model features.

| Split | Dates in 2020, end excluded | Purpose |
|---|---|---|
| Training | Feb 1–Feb 23 | Fit models and scalers |
| Normal development | Feb 23–Mar 1 | CNN early stopping |
| Validation | Mar 1–Jun 15 | Thresholds and selection |
| Test | Jun 15–Sep 2 | Final assessment |

February is assumed normal under UCI's first-month recommendation. Other readings
are unknown unless covered by a failure report. A 24-hour guard band excludes
nearby failures from normal training.

| Event | Reported start | Reported end |
|---|---|---|
| F1 | Apr 18 00:00 | Apr 18 23:59 |
| F2 | May 29 23:30 | May 30 06:00 |
| F3 | Jun 5 10:00 | Jun 7 14:30 |
| F4 | Jul 15 14:30 | Jul 15 19:00 |

F1–F3 are validation events; F4 is the test event. Report ends include the entire
last minute. Timestamps retain the source clock because no timezone is supplied.
The source disagrees about sampling frequency and includes an inconsistent
`30Apr` maintenance note; observed cadence and the listed failure windows are used.

Alerts come from the models, not the dataset. An event is found if an alert falls
inside its window. Unmatched alerts/day counts notifications outside those windows
per observed day outside the windows. They may include unreported problems.
PR-AUC and point precision/recall therefore use imperfect labels. Six-hour advance
warnings are measured separately from detection after onset.

Only one asset and four events are available. Minute sampling and window warm-up
also reduce coverage: July's event has 211 scored minutes out of 271 reported minutes.
Data checks and tables are saved in `artifacts/data/`.
