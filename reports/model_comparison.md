# MetroPT-3 results

Validation selected **CNN autoencoder**. Promotion is **blocked**.

## Validation

| Model | Events detected | Unmatched alerts/day | Threshold |
|---|---:|---:|---:|
| IF raw | 3/3 | 0.961 | 0.735811 |
| IF temporal | 2/3 | 0.910 | 0.675077 |
| CNN autoencoder | 3/3 | 0.628 | 32.518625 |

Selection prefers event recall, then fewer unmatched alerts, then simpler models. Thresholds share a 1-unmatched-alert/day ceiling, 2min persistence and 30min cooldown. Selection and thresholds were saved before test scoring.

## Held-out test

| Metric | IF raw | IF temporal | CNN autoencoder |
|---|---:|---:|---:|
| Event recall | 1.000 | 0.000 | 0.000 |
| Unmatched alerts/day | 0.000 | 0.033 | 0.000 |
| Detection delay, minutes | 138.000 | n/a | n/a |
| PR-AUC | 0.285 | 0.022 | 0.380 |
| Point precision | 0.442 | 0.008 | 0.000 |
| Point recall | 0.346 | 0.005 | 0.000 |
| Inference ms/input row | 0.005 | 0.019 | 0.054 |

Latency includes preprocessing and alert rules for the same CPU batch. Delay is defined only for detected events. Row metrics use incomplete interval-derived labels.

| Model | Event | First alert | Delay, min | Scored coverage | Six-hour prewarning |
|---|---|---|---:|---:|---|
| IF raw | F4 | 2020-07-15T16:48:00 | 138.0 | 0.779 | no |
| IF temporal | F4 | none | n/a | 0.779 | no |
| CNN autoencoder | F4 | none | n/a | 0.779 | no |

## CNN threshold and ranking

The CNN threshold was **32.52**. Its largest test score was **11.29**; the largest score during reported failures was **8.83**. No test score crossed the threshold, so it produced no test alerts.

The CNN's test PR-AUC was 0.380, compared with 0.285 for raw Isolation Forest. Ranking quality and alert performance must be assessed separately. Three validation events provide limited evidence about a later failure. This run does not isolate the causes of the score shift.

The selected candidate remains on hold. Choosing a replacement or lowering its threshold using this test would require a new independent assessment.

## Limits

- One asset, three validation events, and one test event.
- February is assumed normal; other unreported periods remain unknown.
- Unmatched alerts are not verified false positives. PR-AUC and point metrics are proxies.
- Minute snapshots can miss short transients; gaps and warm-up reduce coverage.
- Detection after onset is not advance warning.

Data and split details are in `docs/data.md`. Model settings are in `configs/experiments.yaml`; `uv.lock` pins dependencies.

## Temporal-feature diagnostic

Largest validation score changes under grouped permutation: H1 (0.0214), TP2 (0.0195), DV_pressure (0.0178). These are perturbation sensitivities, not causal importance or a window-length ablation.
