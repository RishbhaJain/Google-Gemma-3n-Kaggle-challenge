"""Data and tokenization integrity checks for the Gemma experiment."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from gemma_experiment.data import sha256_file


@dataclass(frozen=True)
class CompletionEncoding:
    """A causal-LM example whose labels cover only the full completion."""

    input_ids: list[int]
    labels: list[int]
    completion_tokens: int
    removed_prompt_tokens: int


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
