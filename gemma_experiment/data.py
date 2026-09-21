"""Deterministic preparation of conversational prompt-completion splits."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}


def normalize_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Normalize FineTome messages to Hugging Face conversational roles."""
    normalized: list[dict[str, str]] = []
    for message in messages:
        raw_role = message.get("role", message.get("from"))
        raw_content = message.get("content", message.get("value"))
        role = ROLE_MAP.get(str(raw_role), str(raw_role))
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported conversation role: {raw_role!r}")
        if not isinstance(raw_content, str) or not raw_content.strip():
            raise ValueError("Every conversation message must contain non-empty text")
        normalized.append({"role": role, "content": raw_content.strip()})
    return normalized


def to_prompt_completion(row: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Use the final assistant turn as completion and all preceding turns as prompt."""
    raw_messages = row.get("conversations", row.get("messages"))
    if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, str | bytes):
        raise ValueError("Row must contain a conversations or messages sequence")
    messages = normalize_messages(raw_messages)
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        raise ValueError("Conversation must end with an assistant response")
    if not any(message["role"] == "user" for message in messages[:-1]):
        raise ValueError("Conversation prompt must contain at least one user turn")
    return {"prompt": messages[:-1], "completion": [messages[-1]]}


def deterministic_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Return seeded, disjoint train/validation/test splits with stable source IDs."""
    required = train_size + validation_size + test_size
    if required > len(rows):
        raise ValueError(f"Need {required} rows, received {len(rows)}")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    boundaries = (train_size, train_size + validation_size, required)
    groups = {
        "train": indices[: boundaries[0]],
        "validation": indices[boundaries[0] : boundaries[1]],
        "test": indices[boundaries[1] : boundaries[2]],
    }
    result: dict[str, list[dict[str, Any]]] = {}
    for split, split_indices in groups.items():
        examples = []
        for source_index in split_indices:
            example = to_prompt_completion(rows[source_index])
            example["source_index"] = source_index
            examples.append(example)
        result[split] = examples
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
