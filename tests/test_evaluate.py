import pytest

from scripts.evaluate import validate_benchmark_runtime, validate_paired_loss_records


def test_gpu_benchmark_fails_closed_without_cuda():
    with pytest.raises(RuntimeError, match="refusing to publish CPU"):
        validate_benchmark_runtime(cuda_available=False, require_cuda=True)


@pytest.mark.parametrize("cuda_available", [False, True])
def test_runtime_contract_allows_explicit_non_gpu_runs(cuda_available):
    validate_benchmark_runtime(cuda_available=cuda_available, require_cuda=False)


def test_paired_loss_records_require_identical_examples_and_token_counts():
    records = [
        {"example_index": 0, "mean_loss": 1.0, "scored_tokens": 8},
        {"example_index": 1, "mean_loss": 2.0, "scored_tokens": 12},
    ]

    validate_paired_loss_records(records, [dict(record) for record in records])

    with pytest.raises(ValueError, match="token counts differ"):
        validate_paired_loss_records(
            records,
            [records[0], {**records[1], "scored_tokens": 11}],
        )

    with pytest.raises(ValueError, match="indices are not aligned"):
        validate_paired_loss_records(
            records,
            [{**records[0], "example_index": 1}, records[1]],
        )
