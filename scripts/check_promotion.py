#!/usr/bin/env python3
"""Evaluate a completed GPU result against the checked-in promotion policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemma_experiment.promotion import assess_promotion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("results/base_vs_qlora.json"),
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/promotion_policy.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    report = assess_promotion(result, policy)
    print(json.dumps(report, indent=2))
    return 0 if report["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
