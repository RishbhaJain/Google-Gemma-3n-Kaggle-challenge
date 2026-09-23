import json
from pathlib import Path

import pytest

from gemma_experiment.data import sha256_file
from gemma_experiment.integrity import build_completion_encoding, validate_split_checksum


def write_manifest(path: Path, split_path: Path) -> None:
    path.write_text(
        json.dumps({"files": {"test": {"sha256": sha256_file(split_path)}}}),
        encoding="utf-8",
    )


def test_checksum_validation_accepts_manifested_split(tmp_path: Path):
    split_path = tmp_path / "test.jsonl"
    manifest_path = tmp_path / "manifest.json"
    split_path.write_text('{"id": 1}\n', encoding="utf-8")
    write_manifest(manifest_path, split_path)

    assert validate_split_checksum(split_path, manifest_path, "test") == sha256_file(split_path)


def test_checksum_validation_rejects_modified_split(tmp_path: Path):
    split_path = tmp_path / "test.jsonl"
    manifest_path = tmp_path / "manifest.json"
    split_path.write_text('{"id": 1}\n', encoding="utf-8")
    write_manifest(manifest_path, split_path)
    split_path.write_text('{"id": 2}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        validate_split_checksum(split_path, manifest_path, "test")


def test_checksum_validation_requires_manifest_entry(tmp_path: Path):
    split_path = tmp_path / "test.jsonl"
    manifest_path = tmp_path / "manifest.json"
    split_path.write_text("{}\n", encoding="utf-8")
    manifest_path.write_text('{"files": {}}', encoding="utf-8")

    with pytest.raises(ValueError, match="no checksum"):
        validate_split_checksum(split_path, manifest_path, "test")


def test_completion_encoding_preserves_full_target_and_left_truncates_prompt():
    encoded = build_completion_encoding(
        prompt_ids=[1, 2, 3, 4, 5],
        full_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        max_length=5,
    )

    assert encoded.input_ids == [4, 5, 6, 7, 8]
    assert encoded.labels == [-100, -100, 6, 7, 8]
    assert encoded.completion_tokens == 3
    assert encoded.removed_prompt_tokens == 3


def test_completion_encoding_rejects_non_prefix_prompt():
    with pytest.raises(ValueError, match="not a prefix"):
        build_completion_encoding([1, 2], [1, 9, 3], max_length=4)


def test_completion_encoding_rejects_target_that_cannot_fit_intact():
    with pytest.raises(ValueError, match="cannot fit intact"):
        build_completion_encoding([1], [1, 2, 3, 4], max_length=3)
