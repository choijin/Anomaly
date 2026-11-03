# Data and evaluation

[MetroPT-3](https://doi.org/10.24432/C5VW3R) contains 1,516,948 readings from seven
continuous and eight binary compressor sensors. Readings are roughly ten seconds
apart, with 331 gaps longer than 90 seconds despite having no null sensor values.

I keep the last reading each minute without filling gaps. `segment_id` preserves
breaks, and every method is scored at the same endpoints with 60 uninterrupted
minute readings available. Scaling uses training data only; identifiers and
failure-report annotations never become predictors.

| Split | Dates in 2020, end excluded | Purpose |
|---|---|---|
| Training | Feb 1–Feb 23 | Fit models and scalers |
| Normal development | Feb 23–Mar 1 | CNN early stopping |
| Validation | Mar 1–Jun 15 | Choose cutoffs and model |
| Test | Jun 15–Sep 2 | Check the earlier choice |

February is assumed normal under UCI's first-month recommendation. Other readings
remain unknown unless a failure is reported. A 24-hour guard band keeps readings
near reported failures out of normal training.

| Event | Reported start | Reported end | Use |
|---|---|---|---|
| F1 | Apr 18 00:00 | Apr 18 23:59 | Validation |
| F2 | May 29 23:30 | May 30 06:00 | Validation |
| F3 | Jun 5 10:00 | Jun 7 14:30 | Validation |
| F4 | Jul 15 14:30 | Jul 15 19:00 | Test |

I include each report's final minute and keep the original clock because no
timezone is supplied. Where metadata conflicts on frequency or the `30Apr`
maintenance note, I use the observed cadence and listed failure windows.

The models create alerts; the dataset supplies failure windows. An alert inside
a window counts as detection. Unmatched alerts/day counts notifications outside
those windows per observed day outside them. Some may reflect unreported problems,
so this is a false-alarm proxy. Point precision, recall, and PR-AUC also depend
on incomplete report-derived labels. Six-hour advance warnings are separate from
alerts after a failure starts.

One asset and four events limit the comparison. Gaps and warm-up left 211 of
July's 271 reported minutes scorable. Numeric data checks are in `artifacts/data/`.
