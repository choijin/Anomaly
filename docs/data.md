# Data

[MetroPT-3](https://doi.org/10.24432/C5VW3R) contains 1,516,948 readings from seven
continuous and eight binary sensors, sampled about 10 seconds apart. The CSV
has no null readings, although it has 331 gaps longer than 90 seconds.

I take the last reading each minute without filling gaps and preserve breaks
with `segment_id`. I compare all models at the same 60-step
window endpoints, built separately within each split. Each window spans 59 minutes.
I fit scalers on training data only and leave identifiers and report annotations
out of the model features.

| Split | Dates in 2020, end excluded | Purpose |
|---|---|---|
| Training | Feb 1–Feb 23 | Fit models and scalers |
| Normal development | Feb 23–Mar 1 | CNN early stopping |
| Validation | Mar 1–Jun 15 | Thresholds and selection |
| Test | Jun 15–Sep 2 | Final assessment |

I follow UCI's recommendation to treat the first month as normal. Later readings
remain unknown unless a failure was reported, and a 24-hour guard band excludes
readings near failures from normal training.

| Event | Reported start | Reported end |
|---|---|---|
| F1 | Apr 18 00:00 | Apr 18 23:59 |
| F2 | May 29 23:30 | May 30 06:00 |
| F3 | Jun 5 10:00 | Jun 7 14:30 |
| F4 | Jul 15 14:30 | Jul 15 19:00 |

I use F1–F3 for validation and hold F4 for testing, including each report's last
minute. Since no timezone is supplied, I keep the source clock and use observed
cadence and listed failure windows where the frequency or `30Apr` note conflicts.

The models generate alerts; the dataset only supplies failure windows. An alert
inside a window counts as detection. Unmatched alerts/day counts notifications
outside those windows per observed day outside them. Because reports may be
incomplete, those alerts aren't necessarily false alarms, and PR-AUC and point
metrics also use imperfect labels. I measure six-hour advance warnings separately.

This leaves one asset and four events to assess. Sampling and warm-up reduce
coverage too: July has 211 scored minutes out of 271 reported minutes.
Checks and tables are in `artifacts/data/`.
