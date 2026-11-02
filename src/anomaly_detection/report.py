"""Result tables and portable reports for the repository."""

from pathlib import Path
import json
import shutil
import pandas as pd

MODEL_LABELS = {
    "if_raw": "IF raw",
    "if_temporal": "IF temporal",
    "cnn_autoencoder": "CNN autoencoder",
}


def _number(value: float | None, decimals: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{decimals}f}"


def write_comparison_report(summary: dict, output: Path) -> None:
    """Render recorded metrics without changing model selection."""
    rows = []
    for name, result in summary["per_model"].items():
        for split in ["validation", "test"]:
            rows.append(
                {
                    "model": name,
                    "partition": split,
                    **{
                        key: value
                        for key, value in result[f"{split}_metrics"].items()
                        if isinstance(value, (int, float)) or value is None
                    },
                }
            )
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    selected = MODEL_LABELS[summary["selected_model"]]
    blocked = summary["release_blocked"] or not summary["validation_passed"]
    lines = [
        "# Results",
        "",
        f"Validation selected **{selected}**. Release is **{'blocked' if blocked else 'pending review'}**.",
        "",
        "## Validation",
        "",
        "| Model | Events found | Unmatched alerts/day | Threshold |",
        "|---|---:|---:|---:|",
    ]
    for name, result in summary["per_model"].items():
        metrics = result["validation_metrics"]
        lines.append(
            f"| {MODEL_LABELS[name]} | {metrics['events_detected']}/{metrics['events_total']} | {_number(metrics['unmatched_alerts_per_day'])} | {_number(metrics['threshold'], 2)} |"
        )
    alert_rules = json.loads((output / "config.json").read_text())["alerts"]
    lines += [
        "",
        f"Thresholds used validation only, with a {alert_rules['false_alert_budget_per_day']:g}-unmatched-alert/day ceiling, "
        f"{alert_rules['persistence']} persistence, and {alert_rules['cooldown']} cooldown.",
        "",
        "## Test",
        "",
        "| Metric | IF raw | IF temporal | CNN autoencoder |",
        "|---|---:|---:|---:|",
    ]
    test_metrics = [summary["per_model"][name]["test_metrics"] for name in MODEL_LABELS]
    for label, key in [
        ("Event recall", "event_recall"),
        ("Unmatched alerts/day", "unmatched_alerts_per_day"),
        ("Detection delay, minutes", "mean_detection_delay_minutes"),
        ("PR-AUC", "pr_auc"),
        ("Point precision", "precision"),
        ("Point recall", "recall"),
        ("Inference ms/row", "inference_ms_per_row"),
    ]:
        lines.append(
            f"| {label} | "
            + " | ".join(_number(metrics[key]) for metrics in test_metrics)
            + " |"
        )
    coverage = ", ".join(
        f"{event['event_id']}: {event['observed_fraction']:.0%}"
        for event in test_metrics[0]["events"]
    )
    lines += [
        "",
        f"Scored event coverage: {coverage}. Delay is measured after onset for detected events. "
        "Latency is a CPU batch average including preprocessing and alert rules.",
    ]
    cnn = summary["per_model"]["cnn_autoencoder"]
    if "score_max" in cnn["test_metrics"]:
        metrics = cnn["test_metrics"]
        lines += [
            "",
            f"The CNN threshold was **{cnn['validation_metrics']['threshold']:.2f}**, "
            f"against a highest test score of **{metrics['score_max']:.2f}** "
            f"({_number(metrics['failure_score_max'], 2)} during reported failures). "
            + (
                "It produced no test alerts. "
                if metrics["score_max"] < cnn["validation_metrics"]["threshold"]
                else ""
            )
            + f"Its PR-AUC was {_number(metrics['pr_auc'])} versus {_number(test_metrics[0]['pr_auc'])} for raw IF; "
            "ranking and alert-threshold performance are different measures.",
        ]
    lines += [
        "",
        "One asset and four events limit what this comparison can establish. Unreported periods are unknown, "
        "so unmatched alerts and row metrics are proxies. Test results were not used to switch models or retune thresholds.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def publish_results(
    summary: dict, run_dir: Path, report_dir: Path, evaluation_dir: Path
) -> None:
    """Publish reviewable results, excluding machine-specific model paths and IDs."""
    report_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    portable = {
        key: summary[key]
        for key in [
            "selected_model",
            "validation_passed",
            "release_blocked",
            "config_digest",
            "runtime",
            "source",
        ]
    }
    portable["per_model"] = {
        name: {
            key: result[key]
            for key in ["validation_metrics", "test_metrics", "release_blocked"]
        }
        for name, result in summary["per_model"].items()
    }
    (evaluation_dir / "benchmark_results.json").write_text(
        json.dumps(portable, indent=2, allow_nan=False) + "\n"
    )
    shutil.copy2(run_dir / "report.md", report_dir / "model_comparison.md")
    shutil.copy2(run_dir / "comparison.csv", evaluation_dir / "model_comparison.csv")
    for figure in run_dir.glob("*.png"):
        shutil.copy2(figure, figure_dir / figure.name)
    sensitivity = run_dir / "if_temporal" / "validation_score_sensitivity.csv"
    if sensitivity.exists():
        shutil.copy2(sensitivity, evaluation_dir / sensitivity.name)
