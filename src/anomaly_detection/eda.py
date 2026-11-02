"""Data-only inspection, saved before modeling or threshold calibration."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import config_digest
from .data import (
    CONTINUOUS_COLUMNS,
    DIGITAL_COLUMNS,
    SENSOR_COLUMNS,
    CSV_SHA256,
    file_digest,
    load_raw,
    prepare_observations,
)
from .labels import annotate_observations
from .splits import chronological_splits


def highlight_events(axis, events: list[dict]) -> None:
    for event in events:
        axis.axvspan(
            pd.Timestamp(event["start"]),
            pd.Timestamp(event["end"]),
            color="tab:red",
            alpha=0.18,
        )


def prepare_and_inspect(config: dict, output: str | Path = "reports") -> pd.DataFrame:
    """Write measured cadence, gaps, distributions, states, labels, and split protocol."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = load_raw(config["data"]["raw_path"])
    observations = prepare_observations(raw, config)
    annotated = annotate_observations(
        observations,
        config["failure_events"],
        config["trusted_normal_intervals"],
        config["guard_band"],
    )
    partitions = chronological_splits(annotated, config)
    processed = Path(config["data"]["processed_path"])
    processed.parent.mkdir(parents=True, exist_ok=True)
    observations.to_parquet(processed, index=False)
    steps = raw.timestamp.diff().dt.total_seconds().dropna()
    quality = {
        "raw_rows": len(raw),
        "snapshot_rows": len(observations),
        "raw_csv_sha256": file_digest(config["data"]["raw_path"]),
        "expected_csv_sha256": CSV_SHA256,
        "config_digest": config_digest(config),
        "first_timestamp": str(raw.timestamp.min()),
        "last_timestamp": str(raw.timestamp.max()),
        "missing_values": int(raw[SENSOR_COLUMNS].isna().sum().sum()),
        "duplicate_timestamps": int(raw.timestamp.duplicated().sum()),
        "median_sampling_seconds": float(steps.median()),
        "sampling_seconds_percentiles": {
            str(q): float(steps.quantile(q)) for q in [0.01, 0.5, 0.9, 0.99]
        },
        "maximum_gap_seconds": float(steps.max()),
        "raw_gaps_over_90_seconds": int((steps > 90).sum()),
        "snapshot_segments": int(observations.segment_id.nunique()),
        "digital_values": {
            column: sorted(raw[column].unique().tolist()) for column in DIGITAL_COLUMNS
        },
        "constant_signals_in_training": [
            column
            for column in SENSOR_COLUMNS
            if partitions["train"][column].nunique() == 1
        ],
        "timezone": "unspecified source clock; no UTC conversion",
        "sampling_discrepancy": "UCI reports 1 Hz and 0.1 Hz; observed median is 10 seconds",
        "unexpected_endpoint": "CSV ends on 2020-09-01 03:59:50, beyond the February-August description",
        "normality_assumption": "February is assumed normal under UCI's recommendation; other unreported periods are unknown",
        "splits": {
            name: {
                "rows": len(part),
                "start": str(part.timestamp.min()),
                "end": str(part.timestamp.max()),
                "reported_failure_rows": int(part.is_reported_failure.sum()),
            }
            for name, part in partitions.items()
        },
    }
    (output / "data_quality.json").write_text(json.dumps(quality, indent=2) + "\n")
    pd.DataFrame(
        [dict(split=name, **value) for name, value in quality["splits"].items()]
    ).to_csv(output / "split_summary.csv", index=False)
    timeline = []
    for event in config["failure_events"]:
        mask = (observations.timestamp >= pd.Timestamp(event["start"])) & (
            observations.timestamp < pd.Timestamp(event["end"])
        )
        partition = next(
            name
            for name, part in config["splits"].items()
            if pd.Timestamp(part["start"])
            <= pd.Timestamp(event["start"])
            < pd.Timestamp(part["end"])
        )
        timeline.append(
            {**event, "partition": partition, "observed_snapshots": int(mask.sum())}
        )
    pd.DataFrame(timeline).to_csv(output / "failure_timeline.csv", index=False)
    raw[CONTINUOUS_COLUMNS].describe().to_csv(output / "sensor_distributions.csv")
    correlations = observations[SENSOR_COLUMNS].corr()
    correlations.to_csv(output / "sensor_correlations.csv")
    figure, axes = plt.subplots(4, 2, figsize=(13, 10), constrained_layout=True)
    for axis, column in zip(axes.flat, CONTINUOUS_COLUMNS):
        axis.hist(raw[column], bins=70, color="steelblue")
        axis.set(title=column, ylabel="Observations")
    axes.flat[-1].axis("off")
    figure.savefig(output / "sensor_distributions.png", dpi=140)
    plt.close(figure)
    figure, axes = plt.subplots(4, 2, figsize=(12, 10), constrained_layout=True)
    for axis, column in zip(axes.flat, DIGITAL_COLUMNS):
        counts = raw[column].value_counts().reindex([0, 1], fill_value=0)
        axis.bar(["0", "1"], counts, color="steelblue")
        axis.set(title=column, ylabel="Observations")
    figure.savefig(output / "digital_signals.png", dpi=140)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(11, 9), constrained_layout=True)
    heatmap = axis.imshow(correlations, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(15), SENSOR_COLUMNS, rotation=90)
    axis.set_yticks(range(15), SENSOR_COLUMNS)
    figure.colorbar(heatmap, ax=axis, label="Pearson correlation")
    figure.savefig(output / "sensor_correlations.png", dpi=140)
    plt.close(figure)
    chart = (
        observations.set_index("timestamp")[CONTINUOUS_COLUMNS].resample("10min").mean()
    )
    figure, axes = plt.subplots(
        7, 1, figsize=(15, 15), sharex=True, constrained_layout=True
    )
    for axis, column in zip(axes, CONTINUOUS_COLUMNS):
        axis.plot(chart.index, chart[column], linewidth=0.6)
        highlight_events(axis, config["failure_events"])
        axis.set_ylabel(column)
    axes[0].set_title(
        "MetroPT-3 sensor history: reported failure intervals shaded (10-minute display means)"
    )
    figure.savefig(output / "sensor_timeline.png", dpi=140)
    plt.close(figure)
    # These are descriptive electrical-current bands, never normal/failure labels.
    states = observations.assign(
        current_band=pd.cut(
            observations.Motor_current,
            [-np.inf, 1, 6, 8.5, np.inf],
            labels=["near_off", "offload_band", "load_band", "startup_band"],
        )
    )
    states.groupby("current_band", observed=True)[SENSOR_COLUMNS].mean().assign(
        observations=states.current_band.value_counts()
    ).to_csv(output / "operating_states.csv")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for state, rows in states.groupby("current_band", observed=True):
        sample = rows.iloc[:: max(1, len(rows) // 2000)]
        axes[0].scatter(
            sample.Motor_current, sample.TP2, s=2, alpha=0.3, label=str(state)
        )
    axes[0].set(xlabel="Motor current (A)", ylabel="TP2 (bar)")
    axes[0].legend(markerscale=3)
    pd.crosstab(states.COMP, states.DV_eletric).plot.bar(ax=axes[1])
    axes[1].set(title="Compressor control combinations", ylabel="Minute snapshots")
    figure.savefig(output / "operating_states.png", dpi=140)
    plt.close(figure)
    return observations
