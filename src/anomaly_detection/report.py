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
    """Render recorded metrics without training or changing selection."""
    rows = []
    for name, result in summary["per_model"].items():
        for split in ["validation", "test"]:
            metrics = result[f"{split}_metrics"]
            rows.append(
                {
                    "model": name,
                    "partition": split,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if isinstance(value, (int, float)) or value is None
                    },
                }
            )
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    selected = MODEL_LABELS[summary["selected_model"]]
    release_status = (
        "blocked"
        if summary["release_blocked"] or not summary["validation_passed"]
        else "pending review"
    )
    alert_rules = json.loads((output / "config.json").read_text())["alerts"]
    lines = [
        "# MetroPT-3 results",
        "",
        f"Validation selected **{selected}**. Promotion is **{release_status}**.",
        "",
        "## Validation",
        "",
        "| Model | Events detected | Unmatched alerts/day | Threshold |",
        "|---|---:|---:|---:|",
    ]
    for name, result in summary["per_model"].items():
        metrics = result["validation_metrics"]
        lines.append(
            f"| {MODEL_LABELS[name]} | {metrics['events_detected']}/{metrics['events_total']} | "
            f"{_number(metrics['unmatched_alerts_per_day'])} | {_number(metrics['threshold'], 6)} |"
        )
    lines += [
        "",
        "Selection prefers event recall, then fewer unmatched alerts, then simpler models. "
        f"Thresholds share a {alert_rules['false_alert_budget_per_day']:g}-unmatched-alert/day ceiling, "
        f"{alert_rules['persistence']} persistence and {alert_rules['cooldown']} cooldown. "
        "Selection and thresholds were saved before test scoring.",
        "",
        "## Held-out test",
        "",
        "| Metric | IF raw | IF temporal | CNN autoencoder |",
        "|---|---:|---:|---:|",
    ]
    metrics_to_show = [
        ("Event recall", "event_recall"),
        ("Unmatched alerts/day", "unmatched_alerts_per_day"),
        ("Detection delay, minutes", "mean_detection_delay_minutes"),
        ("PR-AUC", "pr_auc"),
        ("Point precision", "precision"),
        ("Point recall", "recall"),
        ("Inference ms/input row", "inference_ms_per_row"),
    ]
    test_metrics = [summary["per_model"][name]["test_metrics"] for name in MODEL_LABELS]
    for label, key in metrics_to_show:
        lines.append(
            f"| {label} | "
            + " | ".join(_number(metrics[key]) for metrics in test_metrics)
            + " |"
        )
    lines += [
        "",
        "Latency includes preprocessing and alert rules for the same CPU batch. "
        "Delay is defined only for detected events. Row metrics use incomplete interval-derived labels.",
        "",
        "| Model | Event | First alert | Delay, min | Scored coverage | Six-hour prewarning |",
        "|---|---|---|---:|---:|---|",
    ]
    for name, result in summary["per_model"].items():
        for event in result["test_metrics"]["events"]:
            lines.append(
                f"| {MODEL_LABELS[name]} | {event['event_id']} | {event['first_alert'] or 'none'} | "
                f"{_number(event['detection_delay_minutes'], 1)} | {_number(event['observed_fraction'])} | "
                f"{'yes' if event['pre_failure_alert'] else 'no'} |"
            )
    cnn = summary["per_model"]["cnn_autoencoder"]
    cnn_test = cnn["test_metrics"]
    if "score_max" in cnn_test:
        lines += [
            "",
            "## CNN threshold and ranking",
            "",
            f"The CNN threshold was **{cnn['validation_metrics']['threshold']:.2f}**. "
            f"Its largest test score was **{cnn_test['score_max']:.2f}**; "
            f"the largest score during reported failures was **{_number(cnn_test['failure_score_max'], 2)}**. "
            + (
                "No test score crossed the threshold, so it produced no test alerts."
                if cnn_test["score_max"] < cnn["validation_metrics"]["threshold"]
                else "Threshold crossings must also satisfy the shared persistence and cooldown policy."
            ),
            "",
            f"The CNN's test PR-AUC was {_number(cnn_test['pr_auc'])}, compared with "
            f"{_number(test_metrics[0]['pr_auc'])} for raw Isolation Forest. "
            "Ranking quality and alert performance must be assessed separately. "
            "Three validation events provide limited evidence about a later failure. "
            "This run does not isolate the causes of the score shift.",
        ]
    if summary["release_blocked"]:
        lines += [
            "",
            "The selected candidate remains on hold. Choosing a replacement or lowering its threshold "
            "using this test would require a new independent assessment.",
        ]
    lines += [
        "",
        "## Limits",
        "",
        "- One asset, three validation events, and one test event.",
        "- February is assumed normal; other unreported periods remain unknown.",
        "- Unmatched alerts are not verified false positives. PR-AUC and point metrics are proxies.",
        "- Minute snapshots can miss short transients; gaps and warm-up reduce coverage.",
        "- Detection after onset is not advance warning.",
        "",
        "Data and split details are in `docs/data.md`. Model settings are in "
        "`configs/experiments.yaml`; `uv.lock` pins dependencies.",
    ]
    sensitivity_path = output / "if_temporal" / "validation_score_sensitivity.csv"
    if sensitivity_path.exists():
        sensitivity = pd.read_csv(sensitivity_path)
        sensors = sensitivity[sensitivity.diagnostic_type == "sensor_group"].nlargest(
            3, "mean_absolute_score_change"
        )
        lines += [
            "",
            "## Temporal-feature diagnostic",
            "",
            "Largest validation score changes under grouped permutation: "
            + ", ".join(
                f"{row.group} ({row.mean_absolute_score_change:.4f})"
                for row in sensors.itertuples()
            )
            + ". These are perturbation sensitivities, not causal importance or a window-length ablation.",
        ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def publish_results(summary: dict, run_dir: Path, report_dir: Path) -> None:
    """Publish reviewable results, excluding machine-specific model paths and IDs."""
    report_dir.mkdir(parents=True, exist_ok=True)
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
    (report_dir / "benchmark_results.json").write_text(
        json.dumps(portable, indent=2, allow_nan=False) + "\n"
    )
    for source, destination in [
        ("report.md", "model_comparison.md"),
        ("comparison.csv", "model_comparison.csv"),
    ]:
        shutil.copy2(run_dir / source, report_dir / destination)
    for figure in run_dir.glob("*.png"):
        shutil.copy2(figure, report_dir / figure.name)
    sensitivity = run_dir / "if_temporal" / "validation_score_sensitivity.csv"
    if sensitivity.exists():
        shutil.copy2(sensitivity, report_dir / sensitivity.name)
