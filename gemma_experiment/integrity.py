"""Data and tokenization integrity checks for the Gemma experiment."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from gemma_experiment.data import sha256_file

FULL_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


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
