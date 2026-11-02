-- Replace YOUR_CATALOG and metropt_dev before running in a SQL warehouse.
-- Unmatched alerts are not confirmed failures. This table contains operational
-- telemetry, not accuracy: maintenance-reviewed labels are required for that.
SELECT
  batch_id,
  get_json_object(metrics_json, '$.start') AS batch_start,
  get_json_object(metrics_json, '$.end') AS batch_end,
  get_json_object(metrics_json, '$.model_version') AS model_version,
  CAST(get_json_object(metrics_json, '$.row_count') AS BIGINT) AS row_count,
  CAST(get_json_object(metrics_json, '$.null_count') AS BIGINT) AS null_count,
  CAST(get_json_object(metrics_json, '$.gap_count') AS BIGINT) AS gap_count,
  CAST(get_json_object(metrics_json, '$.scorable_count') AS BIGINT) AS scorable_count,
  CAST(get_json_object(metrics_json, '$.unscorable_fraction') AS DOUBLE) AS unscorable_fraction,
  CAST(get_json_object(metrics_json, '$.alert_count') AS BIGINT) AS alert_count,
  CAST(get_json_object(metrics_json, '$.alert_fraction') AS DOUBLE) AS alert_fraction,
  CAST(get_json_object(metrics_json, '$.inference_ms_per_row') AS DOUBLE) AS inference_ms_per_row,
  CAST(get_json_object(metrics_json, '$.max_mean_shift') AS DOUBLE) AS max_mean_shift,
  CAST(get_json_object(metrics_json, '$.drift_review') AS BOOLEAN) AS drift_review,
  CAST(get_json_object(metrics_json, '$.quality_passed') AS BOOLEAN) AS quality_passed
FROM YOUR_CATALOG.metropt_dev.anomaly_monitoring
ORDER BY batch_start DESC;
