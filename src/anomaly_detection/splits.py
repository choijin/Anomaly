"""Half-open chronological partitions fixed before any scores are inspected."""

import pandas as pd


def validate_protocol(config: dict) -> None:
    previous_end = None
    for name in ["train", "normal_development", "validation", "test"]:
        interval = config["splits"][name]
        start, end = pd.Timestamp(interval["start"]), pd.Timestamp(interval["end"])
        if start >= end or (previous_end is not None and start < previous_end):
            raise ValueError("Use increasing, non-overlapping chronological splits")
        previous_end = end
    for event in config["failure_events"]:
        start, end = pd.Timestamp(event["start"]), pd.Timestamp(event["end"])
        if start >= end:
            raise ValueError("Failure event must have positive duration")
        owners = [
            name
            for name, part in config["splits"].items()
            if pd.Timestamp(part["start"]) <= start and end <= pd.Timestamp(part["end"])
        ]
        if len(owners) != 1:
            raise ValueError(
                f"Failure event {event['event_id']} must fit wholly inside one split"
            )
        if owners[0] in {"train", "normal_development"}:
            raise ValueError(
                "Reported failures cannot be used for normal-only model fitting"
            )
    if not any(
        pd.Timestamp(event["start"]) >= pd.Timestamp(config["splits"]["test"]["start"])
        for event in config["failure_events"]
    ):
        raise ValueError(
            "Reserve at least one complete failure event for final testing"
        )


def chronological_splits(frame: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    validate_protocol(config)
    result = {}
    for name, interval in config["splits"].items():
        part = (
            frame.loc[
                (frame.timestamp >= pd.Timestamp(interval["start"]))
                & (frame.timestamp < pd.Timestamp(interval["end"]))
            ]
            .copy()
            .reset_index(drop=True)
        )
        if part.empty:
            raise ValueError(f"Empty {name} split")
        # Exclusions become gaps before sequence windows are built.
        if name in {"train", "normal_development"}:
            part = part.loc[part.training_eligible].copy().reset_index(drop=True)
            if part.empty:
                raise ValueError(f"No trusted normal observations in {name}")
        result[name] = part
    return result
