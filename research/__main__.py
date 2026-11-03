"""Download, inspect, and compare models for the MetroPT-3 study."""

import argparse
import json
from pathlib import Path

from anomaly_detection.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m research", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    download.add_argument("--folder", default="data/raw")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", default="configs/experiments.yaml")
    prepare.add_argument("--output", default="artifacts/data")
    prepare.add_argument("--figure-dir", default="reports/figures")
    run = commands.add_parser("run")
    run.add_argument("--config", default="configs/experiments.yaml")
    run.add_argument("--output", default="artifacts/runs/benchmark")
    run.add_argument("--report-dir", default="reports")
    run.add_argument("--evaluation-dir", default="artifacts/evaluation")
    arguments = parser.parse_args()
    if arguments.command == "download":
        from research.download import download_dataset

        print(download_dataset(arguments.folder))
    elif arguments.command == "prepare":
        from research.eda import prepare_and_inspect

        observations = prepare_and_inspect(
            load_config(arguments.config), arguments.output, arguments.figure_dir
        )
        print(
            f"Prepared {len(observations):,} minute snapshots; see {arguments.output}/data_quality.json"
        )
    else:
        import pandas as pd

        from research.benchmark import run_benchmark
        from research.report import publish_results

        config = load_config(arguments.config)
        source = Path(config["data"]["processed_path"])
        if not source.exists():
            raise FileNotFoundError(
                "Run python -m research prepare before comparing models"
            )
        summary = run_benchmark(pd.read_parquet(source), config, Path(arguments.output))
        publish_results(
            summary,
            Path(arguments.output),
            Path(arguments.report_dir),
            Path(arguments.evaluation_dir),
        )
        print(
            json.dumps(
                {
                    key: summary[key]
                    for key in [
                        "selected_model",
                        "validation_passed",
                        "release_blocked",
                        "model_uri",
                    ]
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
