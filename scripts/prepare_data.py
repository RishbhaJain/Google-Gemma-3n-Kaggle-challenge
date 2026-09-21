#!/usr/bin/env python3
"""Download a pinned FineTome revision and create deterministic data splits."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from gemma_experiment.data import deterministic_split, sha256_file, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
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
        "files": files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
