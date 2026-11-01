"""Create the benchmark Delta input once in development, without overwrites."""

from .databricks_helpers import job_parser, spark_session, table_name
from .dataset import load_dataset


def main() -> None:
    arguments = job_parser(__doc__).parse_args()
    if not arguments.schema.endswith("_dev"):
        raise ValueError("Benchmark loading is allowed only in a schema ending in _dev")
    destination = table_name(arguments.catalog, arguments.schema, "metric_readings")
    spark = spark_session()
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {arguments.catalog}.{arguments.schema}")
    readings = load_dataset()
    readings["timestamp"] = readings.timestamp.dt.tz_localize(None)
    spark.createDataFrame(readings).write.format("delta").mode("errorifexists").saveAsTable(destination)
    print(f"Created benchmark table: {destination}")


if __name__ == "__main__":
    main()
