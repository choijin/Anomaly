# Databricks notebook source
# Optional: running this notebook creates/updates a billable serving endpoint.
# Real-time point scores only; persistent per-series alerting stays in the batch job.
# https://databricks-sdk-py.readthedocs.io/en/stable/workspace/serving/serving_endpoints.html
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from mlflow import MlflowClient

dbutils.widgets.text("model_name", "")  # catalog.schema.metric_anomaly
dbutils.widgets.text("endpoint_name", "metric-anomaly")
model_name = dbutils.widgets.get("model_name").strip()
endpoint_name = dbutils.widgets.get("endpoint_name").strip()
if len(model_name.split(".")) != 3 or not endpoint_name:
    raise ValueError("Supply catalog.schema.model and endpoint_name")

mlflow.set_registry_uri("databricks-uc")
registry = MlflowClient()
# Pin this deployment to one resolved champion version, even if the alias changes.
model_version = registry.get_model_version_by_alias(model_name, "champion").version
served_model = ServedEntityInput(
    name="metric-detector", entity_name=model_name, entity_version=str(model_version),
    workload_size="Small", scale_to_zero_enabled=True,
)
workspace = WorkspaceClient()
try:
    workspace.serving_endpoints.get(endpoint_name)
except NotFound:
    endpoint = workspace.serving_endpoints.create_and_wait(
        name=endpoint_name,
        config=EndpointCoreConfigInput(name=endpoint_name, served_entities=[served_model]),
    )
else:
    endpoint = workspace.serving_endpoints.update_config_and_wait(
        name=endpoint_name, served_entities=[served_model],
    )
print(f"Endpoint {endpoint.name} serves {model_name} version {model_version}")
