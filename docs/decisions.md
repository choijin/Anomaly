# Model choices

I started with **Isolation Forest** because it's a simple baseline:
unusual readings tend to be isolated in fewer tree splits. I added rolling
statistics over 1, 5, and 15 minutes to see whether historical context helped,
but they didn't improve this run.

I used a **CNN autoencoder** to learn patterns from 60-step sequences. Its loss
combines standardized MSE for continuous sensors with cross entropy for binary
signals. I used a separate normal period for early stopping and kept failure
reports for threshold selection and evaluation rather than model training.

To choose thresholds, I searched validation-score quantiles under a limit of
one unmatched alert/day, prioritizing event recall and then fewer alerts.
An alert needs two minutes of exceedance, followed by a 30-minute cooldown.

I selected the CNN on validation, but its threshold of 32.52 was well above
July's highest failure score of 8.83. Its test PR-AUC was better than raw IF
(0.380 vs 0.285), so the ranking held up better than the alert threshold.
My selection rule favored fewer alarms over faster detection.

I've kept the candidate on hold. Before changing it, I'd need an independent
assessment. I'd focus on more reviewed failures before adding model complexity.
