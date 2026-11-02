# Model choices

**Isolation Forest** is the baseline. Random splits isolate unusual readings in
fewer steps. The second forest adds rolling means, variability, changes, extremes,
and slopes over 1, 5, and 15 minutes. Those features did not help this run.

**The CNN autoencoder** reconstructs 60-step sensor sequences. Continuous channels
use standardized MSE; binary channels use cross entropy. Early stopping uses a
separate normal period. Training has no failure target, though reported events
are used to choose thresholds and compare models.

Thresholds are validation-score quantiles. Keep candidates below one unmatched
alert/day, prefer event recall, then fewer unmatched alerts. Alerts require two
minutes of exceedance and have a 30-minute cooldown.

The CNN won validation but missed July. Its threshold was 32.52; July's scores
reached only 8.83. It ranked test readings better than raw Isolation Forest
(PR-AUC 0.380 vs 0.285), but its alert threshold did not transfer. Selection also
favored fewer alerts rather than faster detection. Three validation events were
not enough to establish a dependable operating threshold.

The selected candidate stays blocked. Lowering its threshold to catch this known
test event would require a new independent assessment. More assets and reviewed
failure examples would be more useful than adding model complexity.
