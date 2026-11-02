"""Download, inspect, train, and replay MetroPT-3 anomaly detectors."""

import argparse
import json
from pathlib import Path

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    download = subcommands.add_parser("download")
    download.add_argument("--folder", default="data/raw")
    prepare = subcommands.add_parser("prepare")
    prepare.add_argument("--config", default="configs/experiments.yaml")
    prepare.add_argument("--output", default="reports")
    run = subcommands.add_parser("run")
    run.add_argument("--config", default="configs/experiments.yaml")
    run.add_argument("--output", default="reports/runs/benchmark")
    run.add_argument("--report-dir", default="reports")
    predict = subcommands.add_parser("predict")
    for name in ["model", "input", "output"]:
        predict.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    if args.command == "download":
        from .data import download_dataset

        print(download_dataset(args.folder))
    elif args.command == "prepare":
        from .eda import prepare_and_inspect

        observations = prepare_and_inspect(load_config(args.config), args.output)
        print(
            f"Prepared {len(observations):,} causal minute snapshots; see {args.output}/data_quality.json"
        )
    elif args.command == "run":
        import pandas as pd
        from .pipeline import run_experiments

        config = load_config(args.config)
        if not Path(config["data"]["processed_path"]).exists():
            raise FileNotFoundError(
                "Run metropt prepare and foundation tests before model training"
            )
        summary = run_experiments(
            pd.read_parquet(config["data"]["processed_path"]), config, Path(args.output)
        )
        from .report import publish_results

        publish_results(summary, Path(args.output), Path(args.report_dir))
        print(
            json.dumps(
                {
                    key: summary[key]
                    for key in [
                        "selected_model",
                        "validation_passed",
                        "release_blocked",
                        "model_uri",
                        "config_digest",
                    ]
                },
                indent=2,
            )
        )
    else:
        from .predict import predict_file

        predict_file(args.model, args.input, args.output)


if __name__ == "__main__":
    main()
