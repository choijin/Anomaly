"""Result tables and portable reports for the repository."""

import json
import shutil
from pathlib import Path

import pandas as pd

MODEL_LABELS = {
    "if_raw": "Isolation Forest",
    "if_temporal": "Isolation Forest + history",
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
        f"I chose **{selected}** using the earlier failures, then checked it against a later period the model choice hadn't used.",
        "It hasn't passed the checks needed for operational use."
        if blocked
        else "It passed the automated checks and still needs review before operational use.",
        "",
        "## Choosing the model",
        "",
        "I compared how many reported failure periods each model found and how often it raised alerts outside those periods.",
        "These outside alerts are called **unmatched alerts**; without complete maintenance records, I can't assume they're all false alarms.",
        "",
        "| Model | Failures found | Unmatched alerts/day | Alert cutoff |",
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
        f"I allowed at most {alert_rules['false_alert_budget_per_day']:g} unmatched alert per observed day. "
        f"Scores must stay above the cutoff for {alert_rules['persistence']} before an alert; notifications are at least {alert_rules['cooldown']} apart.",
        "A cutoff belongs to its model's score scale, so the cutoff numbers aren't comparable across models.",
        "",
        "## Final check",
        "",
        "| Metric | Isolation Forest | With history | CNN autoencoder |",
        "|---|---:|---:|---:|",
    ]
    test_metrics = [summary["per_model"][name]["test_metrics"] for name in MODEL_LABELS]
    for label, key in [
        ("Fraction of failures found", "event_recall"),
        ("Unmatched alerts/day", "unmatched_alerts_per_day"),
        ("Detection delay, minutes", "mean_detection_delay_minutes"),
        ("Score ranking (PR-AUC)", "pr_auc"),
        ("Inference ms/row", "inference_ms_per_row"),
    ]:
        lines.append(
            f"| {label} | "
            + " | ".join(_number(metrics[key]) for metrics in test_metrics)
            + " |"
        )
    coverage = ", ".join(
        f"{pd.Timestamp(event['start']):%Y-%m-%d} ({event['observed_fraction']:.0%})"
        for event in test_metrics[0]["events"]
    )
    lines += [
        "",
        f"Scorable coverage by failure: {coverage}. Gaps and sequence warm-up explain the missing minutes. "
        "Delay starts at the reported failure onset. Runtime includes preprocessing and alert rules on a local CPU.",
    ]
    cnn = summary["per_model"]["cnn_autoencoder"]
    if "score_max" in cnn["test_metrics"]:
        metrics = cnn["test_metrics"]
        lines += [
            "",
            f"The CNN threshold was **{cnn['validation_metrics']['threshold']:.2f}**, "
            f"while its highest test score was **{metrics['score_max']:.2f}**. "
            + (
                "It produced no test alerts. "
                if metrics["score_max"] < cnn["validation_metrics"]["threshold"]
                else ""
            )
            + "PR-AUC measures whether failure readings receive higher scores than other readings, across possible cutoffs. "
            "Ranking and alerts at a chosen cutoff measure different things.",
        ]
    lines += [
        "",
        "This is a preliminary comparison from one compressor with four reported failures. "
        "I kept the original model choice and cutoffs so the final check remains independent. "
        "Trying a new choice would need more reviewed examples and another independent test. "
        "The numeric results also retain per-reading metrics in `artifacts/evaluation/`.",
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
