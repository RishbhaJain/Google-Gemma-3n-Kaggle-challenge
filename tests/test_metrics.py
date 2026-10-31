import math

import pytest

from gemma_experiment.metrics import (
    exact_match,
    normalize_answer,
    paired_bootstrap_delta,
    paired_weighted_bootstrap_delta,
    paired_weighted_slice_deltas,
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


def test_paired_bootstrap_uses_lower_latency_as_improvement():
    result = paired_bootstrap_delta(
        [1.0, 2.0, 3.0],
        [0.5, 1.5, 2.5],
        samples=200,
        seed=7,
        lower_is_better=True,
    )

    assert result["observed_delta"] == pytest.approx(-0.5)
    assert result["probability_improved"] == 1.0
    assert result["improvement_direction"] == "lower"


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


def test_weighted_bootstrap_matches_token_weighted_loss_delta():
    result = paired_weighted_bootstrap_delta(
        [1.0, 1.0],
        [0.0, 2.0],
        [9, 1],
        samples=500,
        seed=7,
        lower_is_better=True,
    )

    assert result["observed_delta"] == pytest.approx(-0.8)
    assert result["improvement_direction"] == "lower"
    assert result["paired_examples"] == 2
    assert result["paired_tokens"] == 10
    assert result == paired_weighted_bootstrap_delta(
        [1.0, 1.0],
        [0.0, 2.0],
        [9, 1],
        samples=500,
        seed=7,
        lower_is_better=True,
    )


def test_weighted_bootstrap_uses_lower_loss_as_improvement():
    result = paired_weighted_bootstrap_delta(
        [1.0, 2.0, 3.0],
        [0.5, 1.5, 2.5],
        [5, 10, 20],
        samples=200,
        lower_is_better=True,
    )

    assert result["observed_delta"] == pytest.approx(-0.5)
    assert result["probability_improved"] == 1.0


@pytest.mark.parametrize(
    ("baseline", "candidate", "weights", "kwargs", "message"),
    [
        ([], [], [], {}, "same non-zero length"),
        ([1.0], [1.0, 2.0], [1], {}, "same non-zero length"),
        ([1.0], [1.0], [], {}, "weights must have the same length"),
        ([1.0], [1.0], [0], {}, "positive integers"),
        ([1.0], [math.nan], [1], {}, "finite values"),
        ([1.0], [1.0], [1], {"samples": 0}, "at least one"),
        ([1.0], [1.0], [1], {"confidence": 1.0}, "between zero and one"),
    ],
)
def test_weighted_bootstrap_rejects_invalid_inputs(baseline, candidate, weights, kwargs, message):
    with pytest.raises(ValueError, match=message):
        paired_weighted_bootstrap_delta(baseline, candidate, weights, **kwargs)


def test_weighted_length_slices_preserve_pairing_and_token_weights():
    report = paired_weighted_slice_deltas(
        [1.0, 3.0, 2.0, 4.0],
        [0.5, 4.0, 1.5, 5.0],
        [9, 1, 2, 2],
        [32, 100, 200, 700],
        boundaries=[128, 256, 512],
    )

    assert [item["slice"] for item in report["slices"]] == ["1-128", "129-256", "513+"]
    assert report["slices"][0]["base_mean_loss"] == pytest.approx(1.2)
    assert report["slices"][0]["candidate_mean_loss"] == pytest.approx(0.85)
    assert report["slices"][0]["delta_candidate_minus_base"] == pytest.approx(-0.35)
    assert report["slices"][0]["scored_tokens"] == 10
    assert report["largest_delta_slice"] == "513+"
    assert report["largest_delta"] == pytest.approx(1.0)
    assert report["has_observed_regression"] is True


@pytest.mark.parametrize(
    ("baseline", "candidate", "weights", "slice_values", "boundaries", "message"),
    [
        ([], [], [], [], [10], "same non-zero length"),
        ([1.0], [1.0], [], [1], [10], "must match"),
        ([1.0], [1.0], [1], [0], [10], "slice values must be positive"),
        ([1.0], [math.nan], [1], [1], [10], "finite values"),
        ([1.0], [1.0], [1], [1], [10, 10], "unique, strictly increasing"),
    ],
)
def test_weighted_length_slices_reject_invalid_inputs(
    baseline, candidate, weights, slice_values, boundaries, message
):
    with pytest.raises(ValueError, match=message):
        paired_weighted_slice_deltas(
            baseline,
            candidate,
            weights,
            slice_values,
            boundaries=boundaries,
        )
