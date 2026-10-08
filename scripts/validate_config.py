#!/usr/bin/env python3
"""Validate an experiment config without importing training dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemma_experiment.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    return parser.parse_args()


def main() -> None:
    _, report = load_config(parse_args().config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
