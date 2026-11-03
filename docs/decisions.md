# Model choices

I started with **Isolation Forest**, which treats readings as unusual when random
tree splits can isolate them quickly. I added recent averages, variation, and
trends to a second forest to test whether history helped. It didn't improve
this comparison.

The **CNN autoencoder** learns to reconstruct sequences of sensor readings.
Larger reconstruction errors suggest unfamiliar behavior. I balanced squared
error for continuous sensors with cross entropy for binary signals and used a
separate normal period to decide when to stop training.

I chose alert cutoffs on the development data, allowing at most one alert per
observed day outside reported failures. I preferred finding more failure periods,
then fewer unmatched alerts. An alert needs two minutes of sustained exceedance,
with a 30-minute pause between notifications.

That rule chose the CNN, but its cutoff of 32.52 exceeded every score in the
final test (maximum 11.29). It ranked failure readings reasonably well, yet never
raised an alert. My selection rule rewarded fewer alarms without prioritizing
fast detection, and three development events were too little evidence that the
cutoff would work on another failure.

I kept the original comparison so the final test still measures an earlier choice.
The missed failure prevents deployment approval. I'd collect more reviewed failure
examples and define the acceptable detection delay before making another choice
and testing it independently.
