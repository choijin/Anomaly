"""Fit three normal-only detectors; select on validation before opening final test."""

from pathlib import Path
import importlib.metadata
import json
import math
import platform
import time
import joblib
import mlflow
import numpy as np
import pandas as pd

from .config import config_digest
from .data import CSV_SHA256, SENSOR_COLUMNS, validate_observations
from .evaluation import apply_alert_policy, calibrate_threshold, evaluate_predictions
from .features import build_features, sequence_endpoints
from .labels import annotate_observations
from .models.isolation_forest import IsolationForestDetector
from .models.cnn_autoencoder import SequenceAutoencoder
from .operations import passes_validation
from .predict import DetectorArtifact, MODEL_INPUT_COLUMNS
from .splits import chronological_splits
from .tracking import log_model

MODEL_NAMES = ["if_raw", "if_temporal", "cnn_autoencoder"]


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def _numeric_metrics(metrics: dict, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_{name}": float(value)
        for name, value in metrics.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    }


def select_model(validation_results: dict, config: dict) -> str:
    """Only validation metrics are accepted; deterministic simplicity breaks ties."""
    validation_passed = [
        name
        for name, result in validation_results.items()
        if passes_validation(result["validation_metrics"], config["selection"])
    ]
    candidates = validation_passed or list(validation_results)

    def ranking(name):
        metrics = validation_results[name]["validation_metrics"]
        recall, rate = metrics["event_recall"], metrics["unmatched_alerts_per_day"]
        return (
            recall if recall is not None else -1,
            -(rate if rate is not None else np.inf),
            -MODEL_NAMES.index(name),
        )

    return max(candidates, key=ranking)


def _with_labels(scored: pd.DataFrame, part: pd.DataFrame) -> pd.DataFrame:
    return scored.assign(
        is_reported_failure=part.is_reported_failure.to_numpy(),
        event_id=part.event_id.to_numpy(),
        annotation_state=part.annotation_state.to_numpy(),
    )


def _evaluate(scored, config, split):
    interval = config["splits"][split]
    return evaluate_predictions(
        scored,
        config["failure_events"],
        expected_cadence=config["alerts"]["expected_cadence"],
        pre_failure_horizon=config["alerts"]["pre_failure_horizon"],
        period_start=interval["start"],
        period_end=interval["end"],
    )


def run_experiments(
    observations: pd.DataFrame,
    config: dict,
    output_dir: Path,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    source_metadata: dict | None = None,
) -> dict:
    """Train, calibrate, select, then evaluate on the held-out test period.

    All models share eligible endpoints. Scalers fit training only; CNN early
    stopping uses normal development data. Selection is saved before test scoring.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_observations(observations)
    if "segment_id" not in observations:
        raise ValueError(
            "Prepared observations must retain native continuity segment_id"
        )
    annotated = annotate_observations(
        observations,
        config["failure_events"],
        config["trusted_normal_intervals"],
        config["guard_band"],
    )
    partitions = chronological_splits(annotated, config)
    train, development, validation = (
        partitions[name] for name in ["train", "normal_development", "validation"]
    )
    endpoints = {
        name: sequence_endpoints(
            part, config["features"]["sequence_length"], config["data"]["max_gap"]
        )
        for name, part in partitions.items()
    }
    if any(not len(indices) for indices in endpoints.values()):
        raise ValueError("Every split needs a complete gap-safe sequence")
    references = {
        column: {
            "mean": float(train[column].mean()),
            "std": float(train[column].std(ddof=0)),
        }
        for column in SENSOR_COLUMNS
    }
    source = {
        "data_source": "public_benchmark",
        "source_sha256": CSV_SHA256,
        **(source_metadata or {}),
    }
    tracking_uri = tracking_uri or config["tracking"]["local_uri"]
    if tracking_uri.startswith("sqlite:///"):
        Path(tracking_uri.removeprefix("sqlite:///")).parent.mkdir(
            parents=True, exist_ok=True
        )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name or config["tracking"]["experiment_name"])
    results, artifacts = {}, {}
    for name in MODEL_NAMES:
        print(f"Training {name} on February normal data", flush=True)
        folder = output_dir / name
        folder.mkdir(exist_ok=True)
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(
                {
                    "model_name": name,
                    "seed": config["seed"],
                    "config_digest": config_digest(config),
                    "features": "raw_plus_causal_rolls"
                    if name == "if_temporal"
                    else "raw_sensors",
                    "raw_features": len(SENSOR_COLUMNS),
                    "sequence_length": config["features"]["sequence_length"],
                    "sampling": "causal_1minute_last_snapshot",
                    "training_period": json.dumps(config["splits"]["train"]),
                    "normal_development_period": json.dumps(
                        config["splits"]["normal_development"]
                    ),
                    "validation_period": json.dumps(config["splits"]["validation"]),
                    "test_period": json.dumps(config["splits"]["test"]),
                    "source_sha256": source["source_sha256"],
                    "threshold_method": "validation_quantile_grid_at_matched_unmatched_alert_budget",
                    "alert_policy": json.dumps(config["alerts"]),
                    "hyperparameters": json.dumps(
                        config[
                            "autoencoder"
                            if name == "cnn_autoencoder"
                            else "isolation_forest"
                        ]
                    ),
                    "rolling_windows": json.dumps(
                        config["features"]["rolling_windows"]
                        if name == "if_temporal"
                        else []
                    ),
                }
            )
            mlflow.set_tags(
                {
                    "data_source": source["data_source"],
                    "training": "normal_only_no_prediction_target",
                    "selection_partition": "validation",
                    "weak_labels": "failure_reports_not_exhaustive",
                    "config_digest": config_digest(config),
                }
            )
            for context, part in [
                ("training", train),
                ("normal_development", development),
                ("validation", validation),
            ]:
                mlflow.log_input(
                    mlflow.data.from_pandas(
                        part[MODEL_INPUT_COLUMNS],
                        source=source.get("source_table", config["data"]["raw_path"]),
                        name=context,
                    ),
                    context=context,
                )
            started = time.perf_counter()
            if name == "cnn_autoencoder":
                detector = SequenceAutoencoder(
                    config["autoencoder"],
                    config["features"]["sequence_length"],
                    config["seed"],
                )
                detector.fit(
                    train[SENSOR_COLUMNS],
                    endpoints["train"],
                    development[SENSOR_COLUMNS],
                    endpoints["normal_development"],
                )
                for epoch in detector.history:
                    mlflow.log_metrics(
                        {
                            "train_loss": epoch["train_loss"],
                            "normal_development_loss": epoch["development_loss"],
                        },
                        step=int(epoch["epoch"]),
                    )
                _write_json(folder / "training_history.json", detector.history)
                mlflow.log_params(
                    {
                        "best_epoch": detector.best_epoch,
                        "training_windows": detector.training_windows,
                        "development_windows": detector.development_windows,
                    }
                )
            else:
                features = (
                    build_features(
                        train,
                        config["features"]["rolling_windows"],
                        config["data"]["max_gap"],
                    )
                    if name == "if_temporal"
                    else train[SENSOR_COLUMNS]
                )
                detector = IsolationForestDetector(
                    config["isolation_forest"], config["seed"]
                ).fit(features.iloc[endpoints["train"]])
                mlflow.log_params(
                    {
                        "training_rows": detector.training_rows,
                        "model_features": len(detector.feature_names),
                    }
                )
            training_seconds = time.perf_counter() - started
            artifact = DetectorArtifact(name, detector, config, 0.0, references)
            started = time.perf_counter()
            scored = artifact.score_with_details(validation[MODEL_INPUT_COLUMNS])
            elapsed = time.perf_counter() - started
            # Model threshold selection sees validation only, with a shared alert ceiling.
            calibration = calibrate_threshold(
                scored,
                config["failure_events"],
                **config["alerts"],
                period_start=config["splits"]["validation"]["start"],
                period_end=config["splits"]["validation"]["end"],
            )
            artifact.threshold = calibration["threshold"]
            scored = apply_alert_policy(
                scored, artifact.threshold, **calibration["policy"]
            )
            scored = _with_labels(scored, validation)
            if name == "if_temporal":
                from .diagnostics import temporal_score_sensitivity

                temporal_features = build_features(
                    validation,
                    config["features"]["rolling_windows"],
                    config["data"]["max_gap"],
                )
                temporal_score_sensitivity(
                    detector,
                    temporal_features.iloc[endpoints["validation"]],
                    seed=config["seed"],
                ).to_csv(folder / "validation_score_sensitivity.csv", index=False)
            metrics = calibration["metrics"]
            metrics["inference_ms_per_row"] = 1000 * elapsed / len(scored)
            metrics["inference_ms_per_scorable_row"] = (
                1000 * elapsed / int(scored.is_scorable.sum())
            )
            metrics["inference_batch_rows"] = len(scored)
            metrics["training_seconds"] = training_seconds
            metrics["threshold"] = artifact.threshold
            _write_json(folder / "validation_metrics.json", metrics)
            _write_json(folder / "calibration.json", calibration)
            scored.to_parquet(folder / "validation_scores.parquet", index=False)
            mlflow.log_metrics(_numeric_metrics(metrics, "validation"))
            mlflow.log_param("threshold", artifact.threshold)
            model_uri = log_model(artifact, validation)
            model_path = folder / "model.joblib"
            joblib.dump(artifact, model_path)
            mlflow.log_artifacts(str(folder), artifact_path="validation")
            artifacts[name] = artifact
            results[name] = {
                "run_id": run.info.run_id,
                "model_uri": model_uri,
                "validation_metrics": metrics,
                "model_path": str(model_path),
            }
            print(
                f"{name}: validation event recall={metrics['event_recall']:.3f}, unmatched alerts/day={metrics['unmatched_alerts_per_day']:.3f}",
                flush=True,
            )
    selected_model = select_model(results, config)
    validation_passed = passes_validation(
        results[selected_model]["validation_metrics"], config["selection"]
    )
    # Persist selection before scoring the test partition.
    frozen = {
        "selected_model": selected_model,
        "selection_partition": "validation",
        "config_digest": config_digest(config),
        "selection_policy": config["selection"]["policy"],
        "validation_passed": validation_passed,
        "thresholds": {
            name: artifact.threshold for name, artifact in artifacts.items()
        },
        "validation_metrics": {
            name: item["validation_metrics"] for name, item in results.items()
        },
    }
    _write_json(output_dir / "selection.json", frozen)
    print(
        f"Frozen selection: {selected_model}. Evaluating all frozen models on final test.",
        flush=True,
    )
    test = partitions["test"]
    test_frames = {}
    for name, artifact in artifacts.items():
        started = time.perf_counter()
        scored = artifact.score_with_details(test[MODEL_INPUT_COLUMNS])
        elapsed = time.perf_counter() - started
        scored = _with_labels(scored, test)
        metrics = _evaluate(scored, config, "test")
        metrics["score_max"] = float(scored.anomaly_score.max())
        failure_scores = scored.loc[
            scored.is_reported_failure, "anomaly_score"
        ].dropna()
        metrics["failure_score_max"] = (
            float(failure_scores.max()) if not failure_scores.empty else None
        )
        metrics["inference_ms_per_row"] = 1000 * elapsed / len(scored)
        metrics["inference_ms_per_scorable_row"] = (
            1000 * elapsed / int(scored.is_scorable.sum())
        )
        metrics["inference_batch_rows"] = len(scored)
        metrics["threshold"] = artifact.threshold
        results[name]["test_metrics"] = metrics
        test_frames[name] = scored
        results[name]["release_blocked"] = metrics["events_missed"] > 0
        folder = output_dir / name
        _write_json(folder / "test_metrics.json", metrics)
        scored.to_parquet(folder / "test_scores.parquet", index=False)
        with mlflow.start_run(run_id=results[name]["run_id"]):
            mlflow.log_metrics(_numeric_metrics(metrics, "test"))
            mlflow.log_artifacts(str(folder), artifact_path="results")
            mlflow.set_tags(
                {
                    "selected": str(name == selected_model).lower(),
                    "validation_passed": str(
                        passes_validation(
                            results[name]["validation_metrics"], config["selection"]
                        )
                    ).lower(),
                    "release_blocked": str(results[name]["release_blocked"]).lower(),
                    "test_event_recall": str(metrics["event_recall"]),
                }
            )
            # Export the original fitted object and frozen threshold. Never refit on test.
    selected_path = output_dir / "selected_model.joblib"
    joblib.dump(artifacts[selected_model], selected_path)
    restored = joblib.load(selected_path).predict(test[MODEL_INPUT_COLUMNS])
    np.testing.assert_allclose(
        restored.anomaly_score,
        test_frames[selected_model].anomaly_score,
        equal_nan=True,
        rtol=1e-7,
    )
    np.testing.assert_array_equal(
        restored.is_alert, test_frames[selected_model].is_alert
    )
    packaged = mlflow.pyfunc.load_model(results[selected_model]["model_uri"]).predict(
        test[MODEL_INPUT_COLUMNS]
    )
    np.testing.assert_allclose(
        packaged.anomaly_score, restored.anomaly_score, equal_nan=True, rtol=1e-7
    )
    np.testing.assert_array_equal(packaged.is_alert, restored.is_alert)
    runtime = {
        name: importlib.metadata.version(name)
        for name in ["numpy", "pandas", "scikit-learn", "torch", "mlflow"]
    }
    summary = {
        "selected_model": selected_model,
        "selected_run_id": results[selected_model]["run_id"],
        "model_uri": results[selected_model]["model_uri"],
        "selected_model_path": str(selected_path),
        "validation_metrics": results[selected_model]["validation_metrics"],
        "validation_passed": validation_passed,
        "test_metrics": results[selected_model]["test_metrics"],
        "config_digest": config_digest(config),
        "source": source,
        "release_blocked": results[selected_model]["release_blocked"],
        "release_block_reasons": (
            ["Selected candidate missed a reported final-test event"]
            if results[selected_model]["release_blocked"]
            else []
        ),
        "release_review_type": "post_evaluation_engineering_review; does not alter frozen model selection",
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **runtime,
        },
        "per_model": results,
        "model_reload_parity": True,
        "selected_pyfunc_full_test_parity": True,
        "selection_partition": "validation",
        "test_used_for_selection": False,
        "limitations": [
            "Only three validation events and one final test event",
            "Unreported periods are unknown, not certified healthy",
            "One-minute snapshots can miss sub-minute transients",
            "February normality is an explicit provisional assumption",
            "Event onset detection is not advance failure prediction",
            "Normal-only fitting with weak event supervision for calibration/selection",
        ],
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "config.json", config)
    from .visualization import make_model_figures

    make_model_figures(
        test_frames,
        {name: item["test_metrics"] for name, item in results.items()},
        config["failure_events"],
        output_dir,
        autoencoder=artifacts["cnn_autoencoder"].detector,
        test_frame=test,
    )
    from .report import write_comparison_report

    write_comparison_report(summary, output_dir)
    with mlflow.start_run(run_id=results[selected_model]["run_id"]):
        for path in output_dir.iterdir():
            if path.is_file() and path.suffix in {".json", ".png", ".md", ".csv"}:
                mlflow.log_artifact(str(path), artifact_path="comparison")
    return summary
