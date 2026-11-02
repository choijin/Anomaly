# Results

I selected **CNN autoencoder** from the validation results; release is **blocked**.

## Validation

| Model | Events found | Unmatched alerts/day | Threshold |
|---|---:|---:|---:|
| IF raw | 3/3 | 0.961 | 0.74 |
| IF temporal | 2/3 | 0.910 | 0.68 |
| CNN autoencoder | 3/3 | 0.628 | 32.52 |

I chose thresholds on validation data, allowing up to 1 unmatched alert/day with 2min persistence and 30min cooldown.

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

Scored coverage was F4: 78%. I measure delay from failure onset and include preprocessing and alert rules in the CPU batch latency.

The CNN threshold was **32.52**, while its highest test score was **11.29**. It produced no test alerts. Its PR-AUC was 0.380 versus 0.285 for raw IF; I assess ranking separately from alert performance.

Given one asset and four failures, I treat these as preliminary results. Reports may miss problems, so unmatched alerts and row metrics are proxies. I kept model and threshold selection separate from testing.
