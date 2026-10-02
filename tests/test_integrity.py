import json
from pathlib import Path

import pytest

from gemma_experiment.data import sha256_file
from gemma_experiment.integrity import (
    build_completion_encoding,
    pin_adapter_base_revision,
    validate_adapter_base_revision,
    validate_model_revision,
    validate_split_checksum,
)

MODEL_REVISION = "45e9fb1dd0e34db5ff9db1f43a49ac5d8e8b8778"


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


def test_model_revision_requires_full_immutable_commit_sha():
    assert validate_model_revision(MODEL_REVISION) == MODEL_REVISION
    for mutable_or_ambiguous_revision in ("main", "v1", "45e9fb1", "A" * 40):
        with pytest.raises(ValueError, match="full 40-character lowercase"):
            validate_model_revision(mutable_or_ambiguous_revision)


def test_adapter_config_pins_and_validates_base_revision(tmp_path: Path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    config_path = adapter_dir / "adapter_config.json"
    config_path.write_text(
        json.dumps(
            {
                "base_model_name_or_path": "cached/local/model",
                "peft_type": "LORA",
                "revision": None,
            }
        ),
        encoding="utf-8",
    )

    pinned_sha = pin_adapter_base_revision(adapter_dir, "unsloth/gemma-3n-E4B-it", MODEL_REVISION)

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["base_model_name_or_path"] == "unsloth/gemma-3n-E4B-it"
    assert saved["revision"] == MODEL_REVISION
    assert pinned_sha == sha256_file(config_path)
    assert (
        validate_adapter_base_revision(adapter_dir, "unsloth/gemma-3n-E4B-it", MODEL_REVISION)
        == pinned_sha
    )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("base_model_name_or_path", "other/model", "base model mismatch"),
        ("revision", "0" * 40, "base revision mismatch"),
    ],
)
def test_adapter_validation_rejects_drift(tmp_path: Path, field, replacement, message):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    config_path = adapter_dir / "adapter_config.json"
    config = {
        "base_model_name_or_path": "unsloth/gemma-3n-E4B-it",
        "revision": MODEL_REVISION,
    }
    config[field] = replacement
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_adapter_base_revision(adapter_dir, "unsloth/gemma-3n-E4B-it", MODEL_REVISION)


def test_adapter_revision_requires_config_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="adapter config"):
        validate_adapter_base_revision(
            tmp_path / "missing", "unsloth/gemma-3n-E4B-it", MODEL_REVISION
        )


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
