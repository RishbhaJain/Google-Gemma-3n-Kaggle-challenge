"""Deterministic cross-split leakage detection for conversational datasets."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True)
class _PromptRecord:
    split: str
    source_index: int
    fingerprint: str
    shingles: frozenset[tuple[str, ...]]


def normalize_prompt(example: Mapping[str, Any]) -> tuple[str, ...]:
    """Return role-aware normalized prompt tokens without retaining raw text."""
    messages = example.get("prompt")
    if not isinstance(messages, Sequence) or isinstance(messages, str | bytes):
        raise ValueError("Each example must contain a prompt message sequence")

    tokens: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("Every prompt message must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role:
            raise ValueError("Every prompt message must have a non-empty role")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Every prompt message must have non-empty content")
        normalized = unicodedata.normalize("NFKC", content).casefold()
        tokens.append(f"role:{role.casefold()}")
        tokens.extend(TOKEN_PATTERN.findall(normalized))
    if not tokens:
        raise ValueError("Prompt must contain at least one normalized token")
    return tuple(tokens)


def _shingles(tokens: tuple[str, ...], size: int) -> frozenset[tuple[str, ...]]:
    if len(tokens) < size:
        return frozenset((token,) for token in tokens)
    return frozenset(tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1))


def _build_records(
    splits: Mapping[str, Sequence[Mapping[str, Any]]], shingle_size: int
) -> list[_PromptRecord]:
    records: list[_PromptRecord] = []
    for split in sorted(splits):
        examples = splits[split]
        if not isinstance(examples, Sequence) or isinstance(examples, str | bytes):
            raise ValueError(f"Split {split!r} must be an example sequence")
        for position, example in enumerate(examples):
            tokens = normalize_prompt(example)
            fingerprint = hashlib.sha256("\x1f".join(tokens).encode()).hexdigest()
            source_index = example.get("source_index", position)
            if not isinstance(source_index, int):
                raise ValueError("source_index must be an integer when provided")
            records.append(
                _PromptRecord(
                    split=split,
                    source_index=source_index,
                    fingerprint=fingerprint,
                    shingles=_shingles(tokens, shingle_size),
                )
            )
    if len(splits) < 2:
        raise ValueError("Leakage auditing requires at least two splits")
    return records


def _finding(left: _PromptRecord, right: _PromptRecord, similarity: float) -> dict[str, Any]:
    return {
        "left_split": left.split,
        "left_source_index": left.source_index,
        "right_split": right.split,
        "right_source_index": right.source_index,
        "similarity": round(similarity, 6),
    }


def audit_split_leakage(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    near_duplicate_threshold: float = 0.95,
    shingle_size: int = 3,
    max_findings: int = 100,
) -> dict[str, Any]:
    """Find exact and near-duplicate prompts that cross split boundaries.

    Candidate pairs are built with an inverted shingle index, avoiding a full
    quadratic scan for unrelated prompts. Findings contain only split names,
    source indices, and similarity scores, never dataset text.
    """
    if not 0 < near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be in (0, 1]")
    if shingle_size <= 0:
        raise ValueError("shingle_size must be positive")
    if max_findings <= 0:
        raise ValueError("max_findings must be positive")

    records = _build_records(splits, shingle_size)
    exact_findings: list[dict[str, Any]] = []
    near_findings: list[dict[str, Any]] = []
    exact_count = 0
    near_count = 0
    fingerprint_index: dict[str, list[int]] = defaultdict(list)
    shingle_index: dict[tuple[str, ...], list[int]] = defaultdict(list)

    for current_index, current in enumerate(records):
        for previous_index in fingerprint_index[current.fingerprint]:
            previous = records[previous_index]
            if previous.split == current.split:
                continue
            exact_count += 1
            if len(exact_findings) < max_findings:
                exact_findings.append(_finding(previous, current, 1.0))

        candidate_indices: set[int] = set()
        for shingle in current.shingles:
            candidate_indices.update(shingle_index[shingle])
        for previous_index in sorted(candidate_indices):
            previous = records[previous_index]
            if previous.split == current.split or previous.fingerprint == current.fingerprint:
                continue
            union = previous.shingles | current.shingles
            similarity = len(previous.shingles & current.shingles) / len(union)
            if similarity >= near_duplicate_threshold:
                near_count += 1
                if len(near_findings) < max_findings:
                    near_findings.append(_finding(previous, current, similarity))

        fingerprint_index[current.fingerprint].append(current_index)
        for shingle in current.shingles:
            shingle_index[shingle].append(current_index)

    return {
        "schema_version": 1,
        "config": {
            "near_duplicate_threshold": near_duplicate_threshold,
            "shingle_size": shingle_size,
            "max_findings": max_findings,
        },
        "split_rows": {split: len(examples) for split, examples in sorted(splits.items())},
        "summary": {
            "exact_cross_split_pairs": exact_count,
            "near_cross_split_pairs": near_count,
            "passed": exact_count == 0 and near_count == 0,
        },
        "exact_findings": exact_findings,
        "near_findings": near_findings,
        "findings_truncated": exact_count > len(exact_findings) or near_count > len(near_findings),
    }


def require_leakage_free(report: Mapping[str, Any]) -> None:
    """Fail closed when an audit reports any cross-split duplicate."""
    summary = report.get("summary")
    if not isinstance(summary, Mapping) or not isinstance(summary.get("passed"), bool):
        raise ValueError("Invalid leakage audit report")
    if not summary["passed"]:
        raise ValueError(
            "Cross-split leakage detected: "
            f"{summary.get('exact_cross_split_pairs', 0)} exact and "
            f"{summary.get('near_cross_split_pairs', 0)} near-duplicate pairs"
        )
