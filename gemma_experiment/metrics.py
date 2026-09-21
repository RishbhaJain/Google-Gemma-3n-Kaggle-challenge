"""Evaluation metrics shared by scripts and unit tests."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Iterable
from statistics import median


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def summarize_generations(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {
            "exact_match": 0.0,
            "token_f1": 0.0,
            "latency_p50_seconds": 0.0,
            "latency_p95_seconds": 0.0,
            "tokens_per_second": 0.0,
        }
    latencies = [record["latency_seconds"] for record in records]
    total_tokens = sum(record["generated_tokens"] for record in records)
    total_time = sum(latencies)
    return {
        "exact_match": sum(record["exact_match"] for record in records) / len(records),
        "token_f1": sum(record["token_f1"] for record in records) / len(records),
        "latency_p50_seconds": median(latencies),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "tokens_per_second": total_tokens / total_time if total_time else 0.0,
    }


def perplexity(mean_loss: float) -> float:
    return math.exp(min(mean_loss, 50.0))
