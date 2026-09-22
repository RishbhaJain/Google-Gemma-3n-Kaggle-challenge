import math

import pytest

from gemma_experiment.metrics import (
    exact_match,
    normalize_answer,
    percentile,
    perplexity,
    summarize_generations,
    token_f1,
)


def test_normalization_and_exact_match():
    assert normalize_answer("The QUICK, fox!") == "quick fox"
    assert exact_match("An answer.", "answer") == 1.0


def test_token_f1_uses_token_overlap():
    assert token_f1("red blue", "red green") == pytest.approx(0.5)
    assert token_f1("", "") == 1.0
    assert token_f1("red", "blue") == 0.0


def test_percentile_interpolates():
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([1.0, 2.0], 0.95) == pytest.approx(1.95)


def test_generation_summary():
    summary = summarize_generations(
        [
            {"exact_match": 1.0, "token_f1": 1.0, "latency_seconds": 1.0, "generated_tokens": 10.0},
            {"exact_match": 0.0, "token_f1": 0.5, "latency_seconds": 3.0, "generated_tokens": 10.0},
        ]
    )
    assert summary["exact_match"] == 0.5
    assert summary["token_f1"] == 0.75
    assert summary["latency_p50_seconds"] == 2.0
    assert summary["tokens_per_second"] == 5.0


def test_perplexity_is_exponentiated_loss():
    assert perplexity(math.log(10)) == pytest.approx(10.0)
