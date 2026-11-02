# Results

Validation selected **CNN autoencoder**. Release is **blocked**.

## Validation

| Model | Events found | Unmatched alerts/day | Threshold |
|---|---:|---:|---:|
| IF raw | 3/3 | 0.961 | 0.74 |
| IF temporal | 2/3 | 0.910 | 0.68 |
| CNN autoencoder | 3/3 | 0.628 | 32.52 |

Thresholds used validation only, with a 1-unmatched-alert/day ceiling, 2min persistence, and 30min cooldown.

## Test

| Metric | IF raw | IF temporal | CNN autoencoder |
|---|---:|---:|---:|
| Event recall | 1.000 | 0.000 | 0.000 |
| Unmatched alerts/day | 0.000 | 0.033 | 0.000 |
| Detection delay, minutes | 138.000 | n/a | n/a |
| PR-AUC | 0.285 | 0.022 | 0.380 |
| Point precision | 0.442 | 0.008 | 0.000 |
| Point recall | 0.346 | 0.005 | 0.000 |
| Inference ms/row | 0.005 | 0.019 | 0.054 |

Scored event coverage: F4: 78%. Delay is measured after onset for detected events. Latency is a CPU batch average including preprocessing and alert rules.

The CNN threshold was **32.52**, against a highest test score of **11.29** (8.83 during reported failures). It produced no test alerts. Its PR-AUC was 0.380 versus 0.285 for raw IF; ranking and alert-threshold performance are different measures.

One asset and four events limit what this comparison can establish. Unreported periods are unknown, so unmatched alerts and row metrics are proxies. Test results were not used to switch models or retune thresholds.
