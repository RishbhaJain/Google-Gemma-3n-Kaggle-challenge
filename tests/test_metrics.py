import math

import pytest

from gemma_experiment.metrics import (
    exact_match,
    normalize_answer,
    paired_bootstrap_delta,
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
            {
                "exact_match": 1.0,
                "token_f1": 1.0,
                "latency_seconds": 1.0,
                "prompt_tokens": 20.0,
                "generated_tokens": 10.0,
            },
            {
                "exact_match": 0.0,
                "token_f1": 0.5,
                "latency_seconds": 3.0,
                "prompt_tokens": 30.0,
                "generated_tokens": 10.0,
            },
        ]
    )
    assert summary["exact_match"] == 0.5
    assert summary["token_f1"] == 0.75
    assert summary["latency_p50_seconds"] == 2.0
    assert summary["generation_requests"] == 2.0
    assert summary["input_tokens"] == 50.0
    assert summary["output_tokens"] == 20.0
    assert summary["requests_per_second"] == 0.5
    assert summary["tokens_per_second"] == 5.0


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({}, "missing"),
        (
            {
                "exact_match": 1.0,
                "token_f1": 1.0,
                "latency_seconds": 0.0,
                "prompt_tokens": 5.0,
                "generated_tokens": 1.0,
            },
            "latency_seconds must be positive",
        ),
        (
            {
                "exact_match": 1.0,
                "token_f1": 1.0,
                "latency_seconds": 1.0,
                "prompt_tokens": 5.0,
                "generated_tokens": math.nan,
            },
            "must be finite",
        ),
    ],
)
def test_generation_summary_rejects_invalid_benchmark_records(record, message):
    with pytest.raises(ValueError, match=message):
        summarize_generations([record])


def test_perplexity_is_exponentiated_loss():
    assert perplexity(math.log(10)) == pytest.approx(10.0)


def test_paired_bootstrap_is_deterministic_and_preserves_pairing():
    result = paired_bootstrap_delta(
        [0.0, 0.5, 0.5, 1.0],
        [0.5, 1.0, 1.0, 1.0],
        samples=500,
        seed=7,
    )

    assert result["observed_delta"] == pytest.approx(0.375)
    assert result["confidence_interval_low"] >= 0.0
    assert result["confidence_interval_high"] <= 0.5
    assert result["probability_improved"] > 0.95
    assert result["paired_examples"] == 4
    assert result == paired_bootstrap_delta(
        [0.0, 0.5, 0.5, 1.0],
        [0.5, 1.0, 1.0, 1.0],
        samples=500,
        seed=7,
    )


@pytest.mark.parametrize(
    ("baseline", "candidate", "kwargs", "message"),
    [
        ([], [], {}, "same non-zero length"),
        ([1.0], [1.0, 2.0], {}, "same non-zero length"),
        ([1.0], [1.0], {"samples": 0}, "at least one"),
        ([1.0], [1.0], {"confidence": 1.0}, "between zero and one"),
        ([math.nan], [1.0], {}, "finite values"),
    ],
)
def test_paired_bootstrap_rejects_invalid_inputs(baseline, candidate, kwargs, message):
    with pytest.raises(ValueError, match=message):
        paired_bootstrap_delta(baseline, candidate, **kwargs)
