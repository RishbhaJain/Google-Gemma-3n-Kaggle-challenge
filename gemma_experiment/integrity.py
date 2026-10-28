"""Data and tokenization integrity checks for the Gemma experiment."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from gemma_experiment.data import sha256_file

FULL_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
TRAINING_SUMMARY_NAME = "training_summary.json"


@dataclass(frozen=True)
class CompletionEncoding:
    """A causal-LM example whose labels cover only the full completion."""

    input_ids: list[int]
    labels: list[int]
    completion_tokens: int
    removed_prompt_tokens: int


def validate_model_revision(revision: str) -> str:
    """Require an immutable full commit SHA instead of a mutable branch or tag."""
    if not isinstance(revision, str) or not FULL_COMMIT_SHA.fullmatch(revision):
        raise ValueError("Model revision must be a full 40-character lowercase commit SHA")
    return revision


def pin_adapter_base_revision(adapter_dir: Path, base_model: str, revision: str) -> str:
    """Persist the exact base model identity in a saved PEFT adapter config."""
    validate_model_revision(revision)
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing PEFT adapter config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = base_model
    config["revision"] = revision
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return sha256_file(config_path)


def validate_adapter_base_revision(
    adapter_dir: Path, expected_base_model: str, expected_revision: str
) -> str:
    """Fail closed if an adapter could resolve a different base checkpoint."""
    validate_model_revision(expected_revision)
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing PEFT adapter config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("base_model_name_or_path") != expected_base_model:
        raise ValueError(
            "Adapter base model mismatch: expected "
            f"{expected_base_model!r}, found {config.get('base_model_name_or_path')!r}"
        )
    if config.get("revision") != expected_revision:
        raise ValueError(
            "Adapter base revision mismatch: expected "
            f"{expected_revision}, found {config.get('revision')!r}"
        )
    return sha256_file(config_path)


def adapter_artifact_sha256(adapter_dir: Path) -> str:
    """Fingerprint every saved adapter artifact except its self-referential summary."""
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Missing adapter directory: {adapter_dir}")

    files: list[tuple[str, Path]] = []
    for path in adapter_dir.rglob("*"):
        relative_path = path.relative_to(adapter_dir).as_posix()
        if path.is_symlink():
            raise ValueError(f"Adapter artifact must not contain symlinks: {relative_path}")
        if path.is_file() and relative_path != TRAINING_SUMMARY_NAME:
            files.append((relative_path, path))
    if not files:
        raise ValueError(f"Adapter directory contains no artifact files: {adapter_dir}")

    digest = hashlib.sha256()
    for relative_path, path in sorted(files):
        path_bytes = relative_path.encode("utf-8")
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def validate_adapter_artifact(adapter_dir: Path, training_summary_path: Path) -> str:
    """Fail closed if saved adapter files differ from the post-training fingerprint."""
    if not training_summary_path.exists():
        raise FileNotFoundError(f"Missing training summary: {training_summary_path}")
    try:
        summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid training summary JSON: {training_summary_path}") from exc
    if not isinstance(summary, dict):
        raise ValueError(f"Training summary must be a JSON object: {training_summary_path}")

    expected = summary.get("adapter_artifact_sha256")
    if not isinstance(expected, str) or not FULL_SHA256.fullmatch(expected):
        raise ValueError("Training summary has no valid adapter artifact SHA-256")
    actual = adapter_artifact_sha256(adapter_dir)
    if actual != expected:
        raise ValueError(f"Adapter artifact checksum mismatch: expected {expected}, found {actual}")
    return actual


def validate_split_checksum(split_path: Path, manifest_path: Path, split_name: str) -> str:
    """Fail closed when a prepared split no longer matches its manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        expected = manifest["files"][split_name]["sha256"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Manifest has no checksum for split {split_name!r}") from exc
    actual = sha256_file(split_path)
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {split_name}: expected {expected}, found {actual}")
    return actual


def validate_artifact_config(metadata_path: Path, expected_config_sha256: str) -> str:
    """Verify that an artifact belongs to the active experiment contract."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing artifact metadata: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid artifact metadata JSON: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Artifact metadata must be a JSON object: {metadata_path}")
    actual = metadata.get("experiment_config_sha256")
    if actual != expected_config_sha256:
        raise ValueError(
            f"Artifact config mismatch for {metadata_path}: "
            f"expected {expected_config_sha256}, found {actual!r}"
        )
    return sha256_file(metadata_path)


def build_completion_encoding(
    prompt_ids: Sequence[int], full_ids: Sequence[int], max_length: int
) -> CompletionEncoding:
    """Left-truncate only the prompt while retaining every completion token."""
    prompt = list(prompt_ids)
    full = list(full_ids)
    if not prompt:
        raise ValueError("Prompt token sequence must not be empty")
    if full[: len(prompt)] != prompt:
        raise ValueError("Rendered prompt tokens are not a prefix of the full conversation")
    completion = full[len(prompt) :]
    if not completion:
        raise ValueError("Conversation contains no completion tokens")
    if len(completion) >= max_length:
        raise ValueError(
            f"Completion has {len(completion)} tokens and cannot fit intact in "
            f"max_length={max_length} with a causal context token"
        )

    prompt_budget = max_length - len(completion)
    kept_prompt = prompt[-prompt_budget:]
    removed_prompt_tokens = len(prompt) - len(kept_prompt)
    return CompletionEncoding(
        input_ids=kept_prompt + completion,
        labels=[-100] * len(kept_prompt) + completion,
        completion_tokens=len(completion),
        removed_prompt_tokens=removed_prompt_tokens,
    )
