"""Promote an explicitly approved, validated model version; retain rollback metadata."""

import argparse

from .databricks_helpers import table_name


def promote_model(client, source_model: str, source_version: str, target_model: str,
                  approval_ticket: str, allow_benchmark: bool = False):
    from mlflow.exceptions import MlflowException

    for model_name in [source_model, target_model]:
        parts = model_name.split(".")
        if len(parts) != 3:
            raise ValueError("Models need a catalog.schema.name")
        table_name(*parts)
    if not str(source_version).isdigit() or int(source_version) < 1:
        raise ValueError("Choose an explicit positive source version")
    if not approval_ticket.strip():
        raise ValueError("An approval-ticket is required")
    source = client.get_model_version(source_model, source_version)
    if source.tags.get("validation_passed") != "true":
        raise ValueError("Only a validated model version can be promoted")
    data_source = source.tags.get("data_source")
    development_demo = allow_benchmark and target_model.split(".")[1].endswith("_dev")
    if data_source != "reviewed_real" and not (data_source == "public_benchmark" and development_demo):
        raise ValueError("Production promotion requires reviewed_real data; public benchmark demos are development-only")
    try:
        previous = client.get_model_version_by_alias(target_model, "champion")
    except MlflowException as error:
        if error.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        previous = None
    promoted = source if source_model == target_model else client.copy_model_version(
        f"models:/{source_model}/{source_version}", target_model,
    )
    promotion_tags = {
        "approval_ticket": approval_ticket.strip(), "source_model": source_model,
        "source_version": str(source_version), "validation_passed": "true", "data_source": data_source,
    }
    if previous is not None:
        promotion_tags["previous_champion_version"] = str(previous.version)
    for name, value in promotion_tags.items():
        client.set_model_version_tag(target_model, promoted.version, name, value)
    # Alias change is the last operation: failures above leave the champion untouched.
    client.set_registered_model_alias(target_model, "champion", promoted.version)
    return promoted


def main() -> None:
    import mlflow
    from mlflow import MlflowClient
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["source-model", "source-version", "target-model", "approval-ticket"]:
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--allow-benchmark", action="store_true", help="Development schema only")
    arguments = parser.parse_args()
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    promoted = promote_model(
        MlflowClient(), arguments.source_model, arguments.source_version,
        arguments.target_model, arguments.approval_ticket, arguments.allow_benchmark,
    )
    print(f"Champion: models:/{arguments.target_model}/{promoted.version}")


if __name__ == "__main__":
    main()
