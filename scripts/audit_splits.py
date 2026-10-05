#!/usr/bin/env python3
"""Audit prepared train, validation, and test splits for prompt leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemma_experiment.data import read_jsonl
from gemma_experiment.leakage import audit_split_leakage, require_leakage_free


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_config = config["data"]
    audit_config = data_config["leakage_audit"]
    data_dir = Path(data_config["output_dir"])
    splits = {
        split: read_jsonl(data_dir / f"{split}.jsonl") for split in ("train", "validation", "test")
    }
    report = audit_split_leakage(
        splits,
        near_duplicate_threshold=audit_config["near_duplicate_threshold"],
        shingle_size=audit_config["shingle_size"],
        max_findings=audit_config["max_findings"],
    )
    output = args.output or data_dir / "leakage_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    require_leakage_free(report)


if __name__ == "__main__":
    main()
