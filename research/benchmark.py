"""Research-only diagnostics and reporting around the shared training pipeline."""

from pathlib import Path

import joblib
import mlflow
import pandas as pd

from anomaly_detection.data.annotations import annotate_observations
from anomaly_detection.data.features import build_features, sequence_endpoints
from anomaly_detection.data.ingestion import file_digest
from anomaly_detection.training.pipeline import train_and_evaluate
from anomaly_detection.training.splits import chronological_splits
from research.diagnostics import temporal_score_sensitivity
from research.plots import make_model_figures
from research.report import write_comparison_report


def build_analysis(
    summary: dict, observations: pd.DataFrame, config: dict, output_dir: Path
) -> None:
    """Create diagnostics from saved predictions without selecting or refitting models."""
    output_dir = Path(output_dir)
    annotated = annotate_observations(
        observations,
        config["failure_events"],
        config["trusted_normal_intervals"],
        config["guard_band"],
    )
    partitions = chronological_splits(annotated, config)
    validation = partitions["validation"]
    endpoints = sequence_endpoints(
        validation, config["features"]["sequence_length"], config["data"]["max_gap"]
    )
    temporal = joblib.load(output_dir / "if_temporal" / "model.joblib")
    features = build_features(
        validation, config["features"]["rolling_windows"], config["data"]["max_gap"]
    )
    temporal_score_sensitivity(
        temporal.detector, features.iloc[endpoints], seed=config["seed"]
    ).to_csv(
        output_dir / "if_temporal" / "validation_score_sensitivity.csv", index=False
    )
    scores = {
        name: pd.read_parquet(output_dir / name / "test_scores.parquet")
        for name in summary["per_model"]
    }
    autoencoder = joblib.load(output_dir / "cnn_autoencoder" / "model.joblib")
    make_model_figures(
        scores,
        {name: result["test_metrics"] for name, result in summary["per_model"].items()},
        config["failure_events"],
        output_dir,
        autoencoder=autoencoder.detector,
        test_frame=partitions["test"],
    )
    write_comparison_report(summary, output_dir)


def run_benchmark(
    observations: pd.DataFrame,
    config: dict,
    output_dir: Path,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    source_metadata: dict | None = None,
) -> dict:
    provenance = source_metadata or {
        "data_source": "public_benchmark",
        "source_sha256": file_digest(config["data"]["raw_path"]),
        "source_uri": config["data"]["raw_path"],
    }
    summary = train_and_evaluate(
        observations,
        config,
        output_dir,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        source_metadata=provenance,
    )
    build_analysis(summary, observations, config, output_dir)
    with mlflow.start_run(run_id=summary["selected_run_id"]):
        for artifact in Path(output_dir).iterdir():
            if artifact.is_file() and artifact.suffix in {".md", ".png", ".csv"}:
                mlflow.log_artifact(str(artifact), artifact_path="analysis")
    return summary
