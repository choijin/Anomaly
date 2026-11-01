"""Load the public benchmark, compare detectors, package a model, and report."""

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import joblib
import numpy as np

from .dataset import DATASET_NAME, DATA_SOURCE, chronological_split, dataset_metadata, load_dataset
from .detection import AnomalyDetector, apply_alert_policy, evaluate_predictions, monitoring_summary, validate_dataset


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def plot_report(readings, scored_models, detectors, output: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, calibration, evaluation = chronological_split(readings)
    figure, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)
    axes[0].plot(readings.timestamp, readings.value, color="#267b8d", linewidth=0.8)
    axes[0].axvline(calibration.timestamp.min(), color="#777777", linestyle="--", label="Calibration starts")
    axes[0].axvline(evaluation.timestamp.min(), color="#d27700", linestyle="--", label="Evaluation starts")
    axes[0].set_ylabel("CPU utilization (%)")
    axes[0].legend(fontsize=8, loc="upper left")
    for axis, (method, scored) in zip(axes[1:], scored_models.items()):
        axis.plot(scored.timestamp, scored.anomaly_score, linewidth=0.7, color="#244d78", label=method)
        axis.axhline(detectors[method].threshold, linestyle="--", color="#d27700", label="Frozen calibration threshold")
        alerts = scored.loc[scored.alert_started]
        axis.scatter(alerts.timestamp, alerts.anomaly_score, color="#b3282d", marker="^", s=28, label="Alert starts", zorder=4)
        axis.set_ylabel(method.replace("_", " "))
        axis.legend(fontsize=8, loc="upper left")
    for _, event in readings.loc[readings.is_anomaly].groupby("event_id"):
        for axis in axes:
            axis.axvspan(event.timestamp.min(), event.timestamp.max(), color="#dd7373", alpha=0.18)
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Recorded time (stored as UTC by convention); shaded areas are published NAB windows")
    figure.suptitle("Server CPU anomaly detection · real NAB AWS CloudWatch data", fontsize=14)
    figure.savefig(output / "report.png", dpi=140)
    plt.close(figure)


def write_report(metrics: dict, metadata: dict, output: Path) -> None:
    lines = [
        "# Server CPU anomaly detection", "",
        "Real NAB AWS CloudWatch CPU readings: 4,032 observations at five-minute intervals and two published anomaly windows.", "",
        "First three days fit the models; the next day calibrates thresholds; the remaining ten days are held out. "
        "The first two periods contain no published anomaly windows. Each threshold is the 99.5th percentile of its calibration scores. "
        "Labels are excluded from model inputs. No automatic winner selection uses evaluation results.", "",
        "| Method | Point precision | Point recall | Windows detected | Median delay (min) | False alerts / normal series-day |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, values in metrics.items():
        delay = values["median_detection_delay_minutes"]
        lines.append(f"| {method} | {values['point_precision']:.1%} | {values['point_recall']:.1%} | "
                     f"{values['events_detected']}/{values['events_total']} | {delay if delay is not None else 'N/A'} | "
                     f"{values['false_alerts_per_normal_day']:.3f} |")
    lines.extend([
        "", f"Exported model: **{metadata['selected_method']}**, selected by the command argument before evaluation.", "",
        "Both methods fail the example release gates on this benchmark. Candidates remain inspectable in development; promotion blocks validation_passed=false versions.", "",
        "Alerts require two flags among three consecutive readings, separately per series. Missing intervals reset the rule. "
        "A window counts as detected when an alert episode starts inside it. Delay starts at the published window boundary, "
        "for detected windows only; it is not time before failure. False alerts are episode starts outside the published windows, "
        "divided by observed normal series-days.", "",
        "![Recorded values, benchmark windows, and model alerts](report.png)", "",
        "This is a small public benchmark with only two labeled windows. Its windows are broad evaluation annotations, "
        "not proof that every point inside is abnormal or every outside point is normal. Point precision/recall are approximate. "
        "These custom metrics are not the official NAB scoring algorithm. Do not infer production reliability or business impact. "
        "Model changes after examining this holdout need a new final dataset.", "",
        "Sources: [NAB data](https://github.com/numenta/NAB/tree/master/data/realAWSCloudwatch), "
        "[published labels](https://github.com/numenta/NAB/blob/master/labels/combined_windows.json). "
        "Original timestamps have no offset; UTC is a storage convention, not a verified source timezone.",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n")


def run_demo(output: Path, seed: int = 42, track: bool = False, method: str = "robust_zscore") -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    readings = load_dataset()
    validate_dataset(readings)
    readings.to_csv(output / "metric_readings.csv", index=False)
    training, calibration, evaluation = chronological_split(readings)
    if training.is_anomaly.any() or calibration.is_anomaly.any():
        raise ValueError("Background fit/calibration windows overlap published anomaly windows")
    detectors, scored_models, metrics, event_details = {}, {}, {}, {}
    for candidate_method in ["robust_zscore", "isolation_forest"]:
        detector = AnomalyDetector(candidate_method, seed).fit(training).calibrate(calibration)
        predictions = detector.predict(evaluation)
        scored = evaluation.assign(anomaly_score=predictions.anomaly_score, point_flag=predictions.is_anomaly)
        scored = apply_alert_policy(scored)
        detectors[candidate_method], scored_models[candidate_method] = detector, scored
        metrics[candidate_method], event_details[candidate_method] = evaluate_predictions(scored)
        metrics[candidate_method]["threshold"] = detector.threshold
        from .train_model import passes_validation
        metrics[candidate_method]["release_gate_passed"] = passes_validation(metrics[candidate_method])
        scored.to_csv(output / f"scored_{candidate_method}.csv", index=False)
    detector, scored = detectors[method], scored_models[method]
    joblib.dump(detector, output / "model.joblib")
    restored = joblib.load(output / "model.joblib").predict(evaluation)
    np.testing.assert_allclose(restored.anomaly_score, scored.anomaly_score)
    np.testing.assert_array_equal(restored.is_anomaly, scored.point_flag)
    scored.to_csv(output / "scored_test.csv", index=False)
    scored.loc[scored.alert_started].to_csv(output / "alerts.csv", index=False)
    metadata = {
        "dataset_name": DATASET_NAME, "data_source": DATA_SOURCE, "seed": seed,
        "selected_method": method, "data_sha256": dataset_metadata()["raw_csv_sha256"],
        "runtime_versions": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "joblib", "matplotlib"]
        }},
        "split_ranges": {name: {"start": data.timestamp.min().isoformat(), "end": data.timestamp.max().isoformat(), "rows": len(data)}
                         for name, data in zip(["train", "calibration", "test"], [training, calibration, evaluation])},
        "joblib_prediction_parity": "passed_all_test_rows", "mlflow_run_id": None, "mlflow_model_uri": None,
    }
    monitoring = monitoring_summary(training, evaluation)
    monitoring.update(point_flag_rate=float(scored.point_flag.mean()), alert_episodes=int(scored.alert_started.sum()))
    for filename, content in [("metrics.json", metrics), ("events.json", event_details), ("metadata.json", metadata), ("monitoring.json", monitoring)]:
        write_json(output / filename, content)
    plot_report(readings, scored_models, detectors, output)
    write_report(metrics, metadata, output)
    if track:
        from .model_tracking import log_run
        log_run(detector, (training, calibration, evaluation), metrics, metadata, output)
    print(json.dumps(metrics, indent=2))
    print(f"Report: {output / 'report.md'}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--method", choices=["robust_zscore", "isolation_forest"], default="robust_zscore")
    arguments = parser.parse_args()
    run_demo(arguments.output, arguments.seed, arguments.track, arguments.method)
