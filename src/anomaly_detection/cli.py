"""Score prepared sensor history with a packaged anomaly detector."""

import argparse

from anomaly_detection.inference.predictor import predict_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="Score an ordered sensor-history file")
    for name in ["model", "input", "output"]:
        score.add_argument(f"--{name}", required=True)
    arguments = parser.parse_args()
    predict_file(arguments.model, arguments.input, arguments.output)
