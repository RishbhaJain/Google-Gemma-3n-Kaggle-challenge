import pytest

from scripts.evaluate import (
    build_loss_slice_report,
    validate_benchmark_runtime,
    validate_paired_loss_records,
)


def test_gpu_benchmark_fails_closed_without_cuda():
    with pytest.raises(RuntimeError, match="refusing to publish CPU"):
        validate_benchmark_runtime(cuda_available=False, require_cuda=True)


@pytest.mark.parametrize("cuda_available", [False, True])
def test_runtime_contract_allows_explicit_non_gpu_runs(cuda_available):
    validate_benchmark_runtime(cuda_available=cuda_available, require_cuda=False)


def test_paired_loss_records_require_identical_examples_and_token_counts():
    records = [
        {
            "example_index": 0,
            "mean_loss": 1.0,
            "scored_tokens": 8,
            "original_prompt_tokens": 64,
            "completion_tokens": 8,
        },
        {
            "example_index": 1,
            "mean_loss": 2.0,
            "scored_tokens": 12,
            "original_prompt_tokens": 600,
            "completion_tokens": 12,
        },
    ]

    validate_paired_loss_records(records, [dict(record) for record in records])

    with pytest.raises(ValueError, match="scored_tokens differs"):
        validate_paired_loss_records(
            records,
            [records[0], {**records[1], "scored_tokens": 11}],
        )

    with pytest.raises(ValueError, match="indices are not aligned"):
        validate_paired_loss_records(
            records,
            [{**records[0], "example_index": 1}, records[1]],
        )


def test_loss_slice_report_exposes_prompt_and_completion_regressions():
    baseline = [
        {
            "example_index": 0,
            "mean_loss": 1.0,
            "scored_tokens": 8,
            "original_prompt_tokens": 64,
            "completion_tokens": 8,
        },
        {
            "example_index": 1,
            "mean_loss": 2.0,
            "scored_tokens": 80,
            "original_prompt_tokens": 600,
            "completion_tokens": 80,
        },
    ]
    candidate = [
        {**baseline[0], "mean_loss": 0.5},
        {**baseline[1], "mean_loss": 2.5},
    ]

    report = build_loss_slice_report(baseline, candidate)

    assert report["original_prompt_tokens"]["largest_delta_slice"] == "513+"
    assert report["completion_tokens"]["largest_delta_slice"] == "65-128"
    assert report["completion_tokens"]["has_observed_regression"] is True
