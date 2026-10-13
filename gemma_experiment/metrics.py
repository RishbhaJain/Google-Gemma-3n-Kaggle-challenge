"""Evaluation metrics shared by scripts and unit tests."""

from __future__ import annotations

import math
import random
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
            "generation_requests": 0.0,
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "latency_p50_seconds": 0.0,
            "latency_p95_seconds": 0.0,
            "requests_per_second": 0.0,
            "tokens_per_second": 0.0,
        }
    required = {
        "exact_match",
        "token_f1",
        "latency_seconds",
        "prompt_tokens",
        "generated_tokens",
    }
    for index, record in enumerate(records):
        missing = required - record.keys()
        if missing:
            raise ValueError(f"generation record {index} is missing {sorted(missing)}")
        for field in required:
            value = record[field]
            if not math.isfinite(value):
                raise ValueError(f"generation record {index}.{field} must be finite")
        if record["latency_seconds"] <= 0:
            raise ValueError(f"generation record {index}.latency_seconds must be positive")
        if record["prompt_tokens"] < 1:
            raise ValueError(f"generation record {index}.prompt_tokens must be positive")
        if record["generated_tokens"] < 1:
            raise ValueError(f"generation record {index}.generated_tokens must be positive")
    latencies = [record["latency_seconds"] for record in records]
    total_input_tokens = sum(record["prompt_tokens"] for record in records)
    total_tokens = sum(record["generated_tokens"] for record in records)
    total_time = sum(latencies)
    return {
        "exact_match": sum(record["exact_match"] for record in records) / len(records),
        "token_f1": sum(record["token_f1"] for record in records) / len(records),
        "generation_requests": float(len(records)),
        "input_tokens": total_input_tokens,
        "output_tokens": total_tokens,
        "latency_p50_seconds": median(latencies),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "requests_per_second": len(records) / total_time,
        "tokens_per_second": total_tokens / total_time,
    }


def perplexity(mean_loss: float) -> float:
    return math.exp(min(mean_loss, 50.0))


def paired_bootstrap_delta(
    baseline: list[float],
    candidate: list[float],
    *,
    confidence: float = 0.95,
    samples: int = 2_000,
    seed: int = 3407,
) -> dict[str, float | int]:
    """Estimate a paired mean delta and percentile confidence interval.

    Pairing keeps each prompt's base and adapter scores together during
    resampling, so example difficulty is not mistaken for model variance.
    Positive deltas indicate that the candidate performed better.
    """
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("baseline and candidate must have the same non-zero length")
    if samples < 1:
        raise ValueError("samples must be at least one")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if not all(math.isfinite(value) for value in baseline + candidate):
        raise ValueError("bootstrap inputs must contain only finite values")

    paired_deltas = [new - old for old, new in zip(baseline, candidate, strict=True)]
    observed_delta = sum(paired_deltas) / len(paired_deltas)
    generator = random.Random(seed)
    bootstrap_deltas = []
    for _ in range(samples):
        bootstrap_deltas.append(
            sum(generator.choice(paired_deltas) for _ in paired_deltas) / len(paired_deltas)
        )

    alpha = 1 - confidence
    return {
        "observed_delta": observed_delta,
        "confidence_interval_low": percentile(bootstrap_deltas, alpha / 2),
        "confidence_interval_high": percentile(bootstrap_deltas, 1 - alpha / 2),
        "probability_improved": sum(delta > 0 for delta in bootstrap_deltas) / samples,
        "paired_examples": len(paired_deltas),
        "confidence": confidence,
        "bootstrap_samples": samples,
        "seed": seed,
    }


def paired_weighted_bootstrap_delta(
    baseline: list[float],
    candidate: list[float],
    weights: list[int],
    *,
    confidence: float = 0.95,
    samples: int = 2_000,
    seed: int = 3407,
    lower_is_better: bool = False,
) -> dict[str, float | int | str]:
    """Bootstrap a paired, weighted mean delta over evaluation examples.

    Resampling happens at the example level so the base and candidate losses for
    one prompt always move together. Within each resample, token counts weight
    the mean so the point estimate matches completion-level NLL aggregation.
    """
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("baseline and candidate must have the same non-zero length")
    if len(weights) != len(baseline):
        raise ValueError("weights must have the same length as paired values")
    if samples < 1:
        raise ValueError("samples must be at least one")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if not all(math.isfinite(value) for value in baseline + candidate):
        raise ValueError("bootstrap inputs must contain only finite values")
    if any(type(weight) is not int or weight < 1 for weight in weights):
        raise ValueError("weights must be positive integers")

    paired_deltas = [new - old for old, new in zip(baseline, candidate, strict=True)]

    def weighted_delta(indices: list[int]) -> float:
        selected_weight = sum(weights[index] for index in indices)
        return sum(paired_deltas[index] * weights[index] for index in indices) / selected_weight

    indices = list(range(len(paired_deltas)))
    observed_delta = weighted_delta(indices)
    generator = random.Random(seed)
    bootstrap_deltas = []
    for _ in range(samples):
        bootstrap_deltas.append(weighted_delta([generator.choice(indices) for _ in indices]))

    alpha = 1 - confidence
    improved = (
        sum(delta < 0 for delta in bootstrap_deltas)
        if lower_is_better
        else sum(delta > 0 for delta in bootstrap_deltas)
    )
    return {
        "observed_delta": observed_delta,
        "confidence_interval_low": percentile(bootstrap_deltas, alpha / 2),
        "confidence_interval_high": percentile(bootstrap_deltas, 1 - alpha / 2),
        "probability_improved": improved / samples,
        "improvement_direction": "lower" if lower_is_better else "higher",
        "paired_examples": len(paired_deltas),
        "paired_tokens": sum(weights),
        "confidence": confidence,
        "bootstrap_samples": samples,
        "seed": seed,
    }
