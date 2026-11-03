# Results

I chose **CNN autoencoder** using the earlier failures, then checked it against a later period the model choice hadn't used.
It hasn't passed the checks needed for operational use.

## Choosing the model

I compared how many reported failure periods each model found and how often it raised alerts outside those periods.
These outside alerts are called **unmatched alerts**; without complete maintenance records, I can't assume they're all false alarms.

| Model | Failures found | Unmatched alerts/day | Alert cutoff |
|---|---:|---:|---:|
| Isolation Forest | 3/3 | 0.961 | 0.74 |
| Isolation Forest + history | 2/3 | 0.910 | 0.68 |
| CNN autoencoder | 3/3 | 0.628 | 32.52 |

I allowed at most 1 unmatched alert per observed day. Scores must stay above the cutoff for 2min before an alert; notifications are at least 30min apart.
A cutoff belongs to its model's score scale, so the cutoff numbers aren't comparable across models.

## Final check

| Metric | Isolation Forest | With history | CNN autoencoder |
|---|---:|---:|---:|
| Fraction of failures found | 1.000 | 0.000 | 0.000 |
| Unmatched alerts/day | 0.000 | 0.033 | 0.000 |
| Detection delay, minutes | 138.000 | n/a | n/a |
| Score ranking (PR-AUC) | 0.285 | 0.022 | 0.380 |
| Inference ms/row | 0.005 | 0.019 | 0.054 |

Scorable coverage by failure: 2020-07-15 (78%). Gaps and sequence warm-up explain the missing minutes. Delay starts at the reported failure onset. Runtime includes preprocessing and alert rules on a local CPU.

The CNN threshold was **32.52**, while its highest test score was **11.29**. It produced no test alerts. PR-AUC measures whether failure readings receive higher scores than other readings, across possible cutoffs. Ranking and alerts at a chosen cutoff measure different things.

This is a preliminary comparison from one compressor with four reported failures. I kept the original model choice and cutoffs so the final check remains independent. Trying a new choice would need more reviewed examples and another independent test. The numeric results also retain per-reading metrics in `artifacts/evaluation/`.
