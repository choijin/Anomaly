# Public server CPU dataset

Original file: `realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv` from the Numenta Anomaly Benchmark (NAB). It contains 4,032 real AWS CloudWatch CPU-utilization observations at five-minute intervals, spanning 2014-02-14 14:27 through 2014-02-28 14:22.

- `server_cpu.csv`: unmodified original values and offset-free timestamps.
- `anomaly_windows.json`: only this series' two entries from NAB combined window labels.
- `source.json`: original URLs, raw CSV SHA256, provenance and timestamp policy.
- `LICENSE`: original repository license notice.

Sources: [original CSV](https://github.com/numenta/NAB/blob/master/data/realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv), [labels](https://github.com/numenta/NAB/blob/master/labels/combined_windows.json), [dataset description](https://github.com/numenta/NAB/blob/master/data/README.md), [Kaggle listing](https://www.kaggle.com/datasets/boltzmannbrain/nab).

Window labels are broad evaluation annotations, not verified point-by-point incident labels. The project attaches UTC as a storage convention; the original CSV supplies no offset. It does not claim the source timezone is known.

The files are bundled in the Python wheel for offline, credential-free local and Databricks development runs. Public benchmark provenance does not qualify a model for production promotion.
