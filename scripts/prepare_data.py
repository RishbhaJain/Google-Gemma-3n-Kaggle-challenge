#!/usr/bin/env python3
"""Download a pinned FineTome revision and create deterministic data splits."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from gemma_experiment.config import load_config
from gemma_experiment.data import deterministic_split, sha256_file, write_jsonl
from gemma_experiment.leakage import audit_split_leakage, require_leakage_free


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, config_report = load_config(args.config)
    data_config = config["data"]

    from datasets import load_dataset

    dataset = load_dataset(
        data_config["dataset"],
        split=f"{data_config['source_split']}[:{data_config['sample_size']}]",
        revision=data_config["revision"],
    )
    rows = [dataset[index] for index in range(len(dataset))]
    splits = deterministic_split(
        rows,
        train_size=data_config["train_size"],
        validation_size=data_config["validation_size"],
        test_size=data_config["test_size"],
        seed=data_config["seed"],
    )

    output_dir = Path(data_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_config = data_config["leakage_audit"]
    leakage_report = audit_split_leakage(
        splits,
        near_duplicate_threshold=audit_config["near_duplicate_threshold"],
        shingle_size=audit_config["shingle_size"],
        max_findings=audit_config["max_findings"],
    )
    leakage_path = output_dir / "leakage_audit.json"
    leakage_path.write_text(json.dumps(leakage_report, indent=2) + "\n", encoding="utf-8")
    require_leakage_free(leakage_report)

    files = {}
    for split_name, examples in splits.items():
        path = output_dir / f"{split_name}.jsonl"
        count = write_jsonl(path, examples)
        files[split_name] = {
            "path": str(path),
            "rows": count,
            "sha256": sha256_file(path),
        }

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": data_config["dataset"],
        "dataset_revision": data_config["revision"],
        "source_split": data_config["source_split"],
        "sample_size": data_config["sample_size"],
        "seed": data_config["seed"],
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "experiment_config_sha256": config_report["config_sha256"],
        "leakage_audit": {
            "path": str(leakage_path),
            "sha256": sha256_file(leakage_path),
            "summary": leakage_report["summary"],
            "config": leakage_report["config"],
        },
        "files": files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
