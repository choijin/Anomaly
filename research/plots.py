"""Sensor, score, reconstruction, and alert plots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from anomaly_detection.data.features import sequence_endpoints
from anomaly_detection.data.schema import (
    CONTINUOUS_COLUMNS,
    DIGITAL_COLUMNS,
    SENSOR_COLUMNS,
)

MODEL_LABELS = {
    "if_raw": "Isolation Forest: raw",
    "if_temporal": "Isolation Forest: temporal",
    "cnn_autoencoder": "CNN autoencoder",
}
MODEL_COLORS = {
    "if_raw": "#2878a0",
    "if_temporal": "#d88c25",
    "cnn_autoencoder": "#7959a6",
}


def _save(figure, output_dir: Path, name: str) -> None:
    figure.savefig(output_dir / name, dpi=140, bbox_inches="tight")
    plt.close(figure)


def _time_axis(axis) -> None:
    locator = dates.AutoDateLocator(minticks=3, maxticks=7)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(dates.ConciseDateFormatter(locator))
    axis.grid(alpha=0.2)


def _shade_events(
    axis, events: list[dict], start: pd.Timestamp, end: pd.Timestamp
) -> None:
    label_used = False
    for event in events:
        event_start, event_end = (
            pd.Timestamp(event["start"]),
            pd.Timestamp(event["end"]),
        )
        if event_end > start and event_start <= end:
            axis.axvspan(
                max(start, event_start),
                min(end, event_end),
                color="#e5a0a0",
                alpha=0.30,
                label="Reported failure" if not label_used else None,
            )
            label_used = True


def _plot_gap_safe_line(
    axis, frame, value_column, *, color, label, max_points=7000, linewidth=0.7
):
    """Downsample within runs, retaining explicit breaks before any rendering."""
    if frame.empty:
        return
    boundary = frame.timestamp.diff().gt(pd.Timedelta("90s"))
    if "segment_id" in frame:
        boundary |= frame.segment_id.ne(frame.segment_id.shift())
    groups = boundary.cumsum()
    stride = max(1, len(frame) // max_points)
    labeled = False
    for _, run in frame.groupby(groups, sort=False):
        positions = np.unique(np.r_[np.arange(0, len(run), stride), len(run) - 1])
        visible = run.iloc[positions]
        axis.plot(
            visible.timestamp,
            visible[value_column],
            color=color,
            linewidth=linewidth,
            label=label if not labeled else None,
        )
        labeled = True


def _threshold_line(axis, metrics):
    threshold = metrics.get("threshold")
    if threshold is not None and np.isfinite(threshold):
        axis.axhline(
            threshold,
            color="#b64545",
            linestyle="--",
            linewidth=1,
            label=f"Frozen threshold: {threshold:.3g}",
        )


def _score_alert_markers(axis, frame, events):
    alerts = frame.loc[frame.is_alert.fillna(False)]
    matched = _labels(alerts, events)
    axis.scatter(
        alerts.loc[~matched, "timestamp"],
        alerts.loc[~matched, "anomaly_score"],
        marker="^",
        s=22,
        color="#c84949",
        label="Unmatched alert",
        zorder=4,
    )
    axis.scatter(
        alerts.loc[matched, "timestamp"],
        alerts.loc[matched, "anomaly_score"],
        marker="o",
        s=30,
        color="#248050",
        label="Alert during report",
        zorder=4,
    )


def _labels(frame: pd.DataFrame, events: list[dict]) -> np.ndarray:
    if "is_reported_failure" in frame:
        return frame.is_reported_failure.fillna(False).to_numpy(dtype=bool)
    result = np.zeros(len(frame), dtype=bool)
    for event in events:
        result |= frame.timestamp.between(
            pd.Timestamp(event["start"]), pd.Timestamp(event["end"]), inclusive="left"
        ).to_numpy()
    return result


def _number(value, digits: int = 3) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def make_model_figures(
    scored_by_model: dict[str, pd.DataFrame],
    metrics_by_model: dict[str, dict],
    events: list,
    output_dir: Path,
    autoencoder=None,
    test_frame: pd.DataFrame | None = None,
) -> None:
    """Plot recorded predictions; visual subsampling does not affect metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored = {
        name: frame.loc[np.isfinite(frame.anomaly_score)].copy()
        for name, frame in scored_by_model.items()
    }
    scored = {name: frame for name, frame in scored.items() if not frame.empty}
    if not scored:
        return
    with plt.rc_context(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.titlesize": 11,
        }
    ):
        _score_figures(scored, metrics_by_model, events, output_dir)
        _forest_figures(scored, events, output_dir)
        _comparison_figure(scored, metrics_by_model, output_dir)
        _failure_figures(scored, metrics_by_model, events, output_dir, test_frame)
        if autoencoder is not None:
            _autoencoder_figures(scored, events, output_dir, autoencoder, test_frame)


def _score_figures(scored, metrics, events, output_dir):
    figure, axes = plt.subplots(
        len(scored), 1, figsize=(12, 3 * len(scored)), sharex=True, squeeze=False
    )
    for axis, (name, frame) in zip(axes[:, 0], scored.items()):
        _plot_gap_safe_line(
            axis,
            frame,
            "anomaly_score",
            color=MODEL_COLORS.get(name),
            label="Anomaly score",
        )
        _threshold_line(axis, metrics.get(name, {}))
        _score_alert_markers(axis, frame, events)
        _shade_events(axis, events, frame.timestamp.iloc[0], frame.timestamp.iloc[-1])
        axis.set_title(MODEL_LABELS.get(name, name))
        axis.set_ylabel("Anomaly score")
        axis.legend(loc="upper left", fontsize=7, ncol=3)
        _time_axis(axis)
    axes[-1, 0].set_xlabel("Source-clock date (timezone unspecified)")
    figure.suptitle(
        "Frozen-model scores and operational alerts on the final test period", y=1.01
    )
    figure.tight_layout()
    _save(figure, output_dir, "score_timeline.png")

    figure, axes = plt.subplots(
        1, len(scored), figsize=(5 * len(scored), 3.6), squeeze=False
    )
    for axis, (name, frame) in zip(axes[0], scored.items()):
        labels = _labels(frame, events)
        edges = np.linspace(
            frame.anomaly_score.min(), frame.anomaly_score.max() + 1e-12, 45
        )
        for mask, color, label in [
            (~labels, "#6a8090", "Outside reports: unknown"),
            (labels, "#d05a57", "Reported failure"),
        ]:
            if mask.any():
                axis.hist(
                    frame.loc[mask, "anomaly_score"],
                    bins=edges,
                    density=True,
                    alpha=0.55,
                    color=color,
                    label=label,
                )
        axis.set_title(MODEL_LABELS.get(name, name))
        axis.set_xlabel("Anomaly score")
        axis.set_ylabel("Density within annotation group")
        axis.legend(fontsize=7)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    _save(figure, output_dir, "score_distributions.png")

    figure, axis = plt.subplots(figsize=(6.8, 4.8))
    for name, frame in scored.items():
        labels = _labels(frame, events)
        if labels.any() and (~labels).any():
            precision, recall, _ = precision_recall_curve(labels, frame.anomaly_score)
            value = metrics.get(name, {}).get("pr_auc")
            axis.plot(
                recall,
                precision,
                label=f"{MODEL_LABELS.get(name, name)} (PR-AUC {_number(value)})",
                color=MODEL_COLORS.get(name),
            )
    reference = next(iter(scored.values()))
    prevalence = _labels(reference, events).mean()
    axis.axhline(
        prevalence,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label=f"Interval-label prevalence: {prevalence:.3f}",
    )
    axis.set(
        xlabel="Recall against interval-derived labels",
        ylabel="Precision against interval-derived labels",
        xlim=(0, 1),
        ylim=(0, 1.02),
        title="Secondary ranking metric: incomplete proxy labels",
    )
    axis.legend(fontsize=7)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    _save(figure, output_dir, "precision_recall.png")


def _forest_figures(scored, events, output_dir):
    forest_frames = [
        (name, frame)
        for name, frame in scored.items()
        if "average_path_length" in frame
    ]
    if forest_frames:
        figure, axes = plt.subplots(
            1, len(forest_frames), figsize=(5.2 * len(forest_frames), 4), squeeze=False
        )
        for axis, (name, frame) in zip(axes[0], forest_frames):
            visible = frame.iloc[:: max(1, len(frame) // 5000)]
            labels = _labels(visible, events)
            axis.scatter(
                visible.average_path_length,
                visible.anomaly_score,
                c=np.where(labels, "#d05a57", "#6a8090"),
                s=4,
                alpha=0.5,
            )
            axis.set(
                xlabel="Mean corrected isolation-tree path length",
                ylabel="Anomaly score",
                title=MODEL_LABELS.get(name, name),
            )
            axis.grid(alpha=0.2)
        figure.suptitle("Shorter effective paths produce higher anomaly scores", y=1.02)
        figure.tight_layout()
        _save(figure, output_dir, "isolation_path_lengths.png")
    if "if_raw" in scored and "if_temporal" in scored:
        joined = scored["if_raw"][["timestamp", "anomaly_score"]].merge(
            scored["if_temporal"][["timestamp", "anomaly_score"]],
            on="timestamp",
            suffixes=("_raw", "_temporal"),
            validate="one_to_one",
        )
        visible = joined.iloc[:: max(1, len(joined) // 6000)]
        failure = _labels(visible, events)
        figure, axis = plt.subplots(figsize=(6, 4.6))
        for mask, color, label in [
            (~failure, "#6a8090", "Outside reports: unknown"),
            (failure, "#d05a57", "Reported failure"),
        ]:
            axis.scatter(
                visible.loc[mask, "anomaly_score_raw"],
                visible.loc[mask, "anomaly_score_temporal"],
                s=6,
                alpha=0.5,
                color=color,
                label=label,
            )
        axis.set(
            xlabel="Isolation Forest raw-feature score",
            ylabel="Isolation Forest temporal-feature score",
            title="Same held-out timestamps, different sensor context",
        )
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
        figure.tight_layout()
        _save(figure, output_dir, "raw_vs_temporal_scores.png")


def _comparison_figure(scored, metrics, output_dir):
    columns = [
        "Event\nrecall",
        "Unmatched\nalerts/day",
        "Detection\ndelay (min)",
        "PR-AUC\n(proxy)",
        "Precision\n(proxy)",
        "Recall\n(proxy)",
        "Inference\nms/row",
    ]
    keys = [
        "event_recall",
        "unmatched_alerts_per_day",
        "mean_detection_delay_minutes",
        "pr_auc",
        "precision",
        "recall",
        "inference_ms_per_row",
    ]
    rows, names = [], []
    for name in scored:
        values = metrics.get(name, {})
        rows.append([_number(values.get(key), 3) for key in keys])
        names.append(MODEL_LABELS.get(name, name))
    figure, axis = plt.subplots(figsize=(12, 2.8))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        rowLabels=names,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    axis.set_title(
        "Actual final-test comparison (selection used validation only)", pad=24
    )
    figure.text(
        0.1,
        0.02,
        "Delay is conditional on detection; — means undefined. Unmatched alerts are not verified false positives. Latency is batch-amortized CPU time.",
        fontsize=8,
    )
    _save(figure, output_dir, "model_comparison.png")


def _autoencoder_figures(scored, events, output_dir, autoencoder, test_frame):
    history = pd.DataFrame(autoencoder.history)
    if not history.empty:
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(history.epoch, history.train_loss, label="Normal training windows")
        axis.plot(
            history.epoch,
            history.development_loss,
            label="Separate normal development windows",
        )
        axis.axvline(
            autoencoder.best_epoch,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            label="Restored best epoch",
        )
        axis.set(
            xlabel="Epoch",
            ylabel="Weighted MSE/BCE reconstruction loss",
            title="CNN autoencoder training and early stopping",
        )
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
        figure.tight_layout()
        _save(figure, output_dir, "autoencoder_training.png")
    frame = scored.get("cnn_autoencoder")
    if frame is not None:
        failure = _labels(frame, events)
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        for axis, sensors, title in [
            (axes[0], CONTINUOUS_COLUMNS, "Continuous channels: scaled MSE"),
            (axes[1], DIGITAL_COLUMNS, "Digital channels: binary cross entropy"),
        ]:
            available = [sensor for sensor in sensors if f"error_{sensor}" in frame]
            for offset, mask, color, label in [
                (-0.18, ~failure, "#6a8090", "Outside reports: unknown"),
                (0.18, failure, "#d05a57", "Reported failure"),
            ]:
                if available and mask.any():
                    averages = [
                        frame.loc[mask, f"error_{sensor}"].mean()
                        for sensor in available
                    ]
                    axis.bar(
                        np.arange(len(available)) + offset,
                        averages,
                        width=0.36,
                        color=color,
                        label=label,
                    )
            axis.set_xticks(
                np.arange(len(available)), available, rotation=45, ha="right"
            )
            axis.set_title(title)
            axis.set_ylabel("Mean reconstruction error")
            axis.legend(fontsize=7)
            axis.grid(axis="y", alpha=0.2)
        figure.suptitle(
            "Channel diagnostics: compare within loss groups; scales differ", y=1.02
        )
        figure.tight_layout()
        _save(figure, output_dir, "autoencoder_feature_errors.png")
    if test_frame is None or test_frame.empty:
        return
    endpoints = sequence_endpoints(
        test_frame, autoencoder.sequence_length, max_gap="90s"
    )
    if not len(endpoints):
        return
    eligible = test_frame.iloc[endpoints]
    reported = _labels(eligible, events)
    end = (
        int(endpoints[np.flatnonzero(reported)[0]])
        if reported.any()
        else int(endpoints[0])
    )
    window = test_frame.iloc[end - autoencoder.sequence_length + 1 : end + 1]
    reconstruction = autoencoder.reconstruct(window[SENSOR_COLUMNS].to_numpy())
    figure, axes = plt.subplots(
        len(CONTINUOUS_COLUMNS), 1, figsize=(11, 13), sharex=True
    )
    for index, (axis, sensor) in enumerate(zip(axes, CONTINUOUS_COLUMNS)):
        axis.plot(
            window.timestamp,
            window[sensor],
            label="Observed",
            color="#2878a0",
            linewidth=1,
        )
        axis.plot(
            window.timestamp,
            reconstruction[:, index],
            label="Reconstructed",
            color="#d88c25",
            linewidth=1,
        )
        _shade_events(axis, events, window.timestamp.iloc[0], window.timestamp.iloc[-1])
        axis.set_ylabel(sensor)
        _time_axis(axis)
        if index == 0:
            axis.legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel("Source-clock date (timezone unspecified)")
    figure.suptitle(
        "Observed versus reconstructed sensors: first eligible failure window", y=1.01
    )
    figure.tight_layout()
    _save(figure, output_dir, "autoencoder_reconstruction.png")


def _failure_figures(scored, metrics, events, output_dir, test_frame):
    """Zoom onto held-out reports without changing frozen scores or thresholds."""
    start = min(frame.timestamp.iloc[0] for frame in scored.values())
    end = max(frame.timestamp.iloc[-1] for frame in scored.values())
    held_out = [
        event
        for event in events
        if pd.Timestamp(event["start"]) <= end and pd.Timestamp(event["end"]) > start
    ]
    for event_number, event in enumerate(held_out):
        left = pd.Timestamp(event["start"]) - pd.Timedelta("6h")
        right = pd.Timestamp(event["end"]) + pd.Timedelta("6h")
        suffix = "" if len(held_out) == 1 else f"_{event_number + 1}"
        figure, axes = plt.subplots(
            len(scored), 1, figsize=(12, 3 * len(scored)), sharex=True, squeeze=False
        )
        for axis, (name, frame) in zip(axes[:, 0], scored.items()):
            zoom = frame.loc[frame.timestamp.between(left, right)]
            _plot_gap_safe_line(
                axis,
                zoom,
                "anomaly_score",
                color=MODEL_COLORS.get(name),
                label="Anomaly score",
                max_points=3000,
            )
            _threshold_line(axis, metrics.get(name, {}))
            _score_alert_markers(axis, zoom, [event])
            _shade_events(axis, [event], left, right)
            detail = next(
                (
                    item
                    for item in metrics.get(name, {}).get("events", [])
                    if item["event_id"] == event["event_id"]
                ),
                None,
            )
            outcome = "No event-level result"
            if detail is not None:
                if detail["detected"]:
                    outcome = f"Detected {detail['detection_delay_minutes']:.0f} min after reported start"
                else:
                    outcome = "Missed reported failure"
                coverage = detail.get("observed_fraction")
                if coverage is not None:
                    outcome += f"; {coverage:.0%} scored coverage"
            axis.set_title(f"{MODEL_LABELS.get(name, name)} — {outcome}")
            axis.set_ylabel("Anomaly score")
            axis.set_xlim(left, right)
            axis.legend(loc="upper left", fontsize=7, ncol=3)
            _time_axis(axis)
        axes[-1, 0].set_xlabel(
            "Source-clock time; six hours before and after the reported interval"
        )
        figure.suptitle(
            f"Held-out {event['event_id']}: frozen anomaly scores and alert outcomes",
            y=1.01,
        )
        figure.tight_layout()
        _save(figure, output_dir, f"failure_zoom{suffix}.png")

        if test_frame is None or test_frame.empty:
            continue
        sensor_zoom = test_frame.loc[test_frame.timestamp.between(left, right)]
        if sensor_zoom.empty:
            continue
        sensors = ["TP2", "Motor_current", "Oil_temperature"]
        figure, axes = plt.subplots(len(sensors), 1, figsize=(12, 8), sharex=True)
        markers = {"if_raw": "o", "if_temporal": "^", "cnn_autoencoder": "s"}
        for axis, sensor in zip(axes, sensors):
            _plot_gap_safe_line(
                axis,
                sensor_zoom,
                sensor,
                color="#6a8090",
                label="Observed sensor",
                max_points=3000,
            )
            _shade_events(axis, [event], left, right)
            for name, frame in scored.items():
                alerts = frame.loc[
                    frame.timestamp.between(left, right) & frame.is_alert.fillna(False),
                    ["timestamp"],
                ]
                points = alerts.merge(
                    sensor_zoom[["timestamp", sensor]],
                    on="timestamp",
                    how="inner",
                    validate="one_to_one",
                )
                # An empty scatter retains the legend key, showing detectors with no alerts.
                axis.scatter(
                    points.timestamp,
                    points[sensor],
                    marker=markers.get(name, "o"),
                    s=36,
                    edgecolor="black",
                    linewidth=0.4,
                    color=MODEL_COLORS.get(name),
                    label=f"{MODEL_LABELS.get(name, name)} alerts ({len(points)})",
                    zorder=4,
                )
            axis.set_ylabel(
                {
                    "TP2": "TP2 (bar)",
                    "Motor_current": "Motor current (A)",
                    "Oil_temperature": "Oil temperature (°C)",
                }[sensor]
            )
            axis.set_xlim(left, right)
            _time_axis(axis)
        axes[0].legend(loc="upper left", fontsize=7, ncol=3)
        axes[-1].set_xlabel("Source-clock time (timezone unspecified)")
        figure.suptitle(
            f"Held-out {event['event_id']}: sensor readings with operational alerts",
            y=1.01,
        )
        figure.tight_layout()
        _save(figure, output_dir, f"failure_sensor_alerts{suffix}.png")
