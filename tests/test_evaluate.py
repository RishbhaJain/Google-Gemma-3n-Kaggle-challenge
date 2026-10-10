import pytest

from scripts.evaluate import validate_benchmark_runtime


def test_gpu_benchmark_fails_closed_without_cuda():
    with pytest.raises(RuntimeError, match="refusing to publish CPU"):
        validate_benchmark_runtime(cuda_available=False, require_cuda=True)


@pytest.mark.parametrize("cuda_available", [False, True])
def test_runtime_contract_allows_explicit_non_gpu_runs(cuda_available):
    validate_benchmark_runtime(cuda_available=cuda_available, require_cuda=False)
