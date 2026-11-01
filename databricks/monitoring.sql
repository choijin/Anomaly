-- Select the environment's catalog/schema in the SQL editor first.

-- Operational health and exact model/data versions for each batch.
SELECT
  window_start,
  window_end,
  model_name,
  model_version,
  source_table_version,
  row_count,
  point_flag_rate,
  alert_count,
  max_mean_shift,
  data_lag_minutes,
  quality_status,
  drift_review,
  status
FROM batch_monitoring
ORDER BY window_start DESC
LIMIT 30;

-- Alert volume by series and UTC day.
SELECT
  DATE(timestamp) AS alert_date,
  series_id,
  COUNT(*) AS alert_episodes
FROM metric_alerts
WHERE alert_started
GROUP BY DATE(timestamp), series_id
ORDER BY alert_date DESC, series_id;

-- Ingestion freshness is separate from a backfill's scoring window.
SELECT
  MAX(timestamp) AS latest_metric_timestamp,
  TIMESTAMPDIFF(MINUTE, MAX(timestamp), CURRENT_TIMESTAMP()) AS ingestion_lag_minutes
FROM metric_readings;
