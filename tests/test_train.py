import pytest

from scripts.train import capture_gpu_memory_baseline, summarize_training_gpu_memory


class FakeCuda:
    def __init__(
        self,
        *,
        available: bool,
        allocated: int = 0,
        reserved: int = 0,
        peak_allocated: int = 0,
        peak_reserved: int = 0,
    ):
        self.available = available
        self.allocated = allocated
        self.reserved = reserved
        self.peak_allocated = peak_allocated
        self.peak_reserved = peak_reserved

    def is_available(self):
        return self.available

    def memory_allocated(self):
        return self.allocated

    def memory_reserved(self):
        return self.reserved

    def max_memory_allocated(self):
        return self.peak_allocated

    def max_memory_reserved(self):
        return self.peak_reserved


def test_training_gpu_memory_separates_baseline_peak_and_incremental_pressure():
    gibibyte = 1024**3
    cuda = FakeCuda(
        available=True,
        allocated=4 * gibibyte,
        reserved=5 * gibibyte,
        peak_allocated=7 * gibibyte,
        peak_reserved=9 * gibibyte,
    )

    baseline = capture_gpu_memory_baseline(cuda)
    summary = summarize_training_gpu_memory(cuda, baseline)

    assert summary == {
        "baseline_allocated_gib": 4.0,
        "baseline_reserved_gib": 5.0,
        "peak_allocated_gib": 7.0,
        "peak_reserved_gib": 9.0,
        "peak_allocator_overhead_gib": 2.0,
        "incremental_peak_allocated_gib": 3.0,
        "incremental_peak_reserved_gib": 4.0,
    }


def test_training_gpu_memory_reports_zero_without_cuda():
    cuda = FakeCuda(available=False)

    assert capture_gpu_memory_baseline(cuda) == {
        "allocated_bytes": 0,
        "reserved_bytes": 0,
    }
    assert summarize_training_gpu_memory(cuda, capture_gpu_memory_baseline(cuda)) == {
        "baseline_allocated_gib": 0.0,
        "baseline_reserved_gib": 0.0,
        "peak_allocated_gib": 0.0,
        "peak_reserved_gib": 0.0,
        "peak_allocator_overhead_gib": 0.0,
        "incremental_peak_allocated_gib": 0.0,
        "incremental_peak_reserved_gib": 0.0,
    }


@pytest.mark.parametrize(
    "cuda,baseline,message",
    [
        (
            FakeCuda(available=True, allocated=2, reserved=1),
            None,
            "Invalid CUDA baseline counters",
        ),
        (
            FakeCuda(available=True, peak_allocated=5, peak_reserved=4),
            {"allocated_bytes": 1, "reserved_bytes": 2},
            "reserved bytes must be at least allocated",
        ),
        (
            FakeCuda(available=True, peak_allocated=3, peak_reserved=4),
            {"allocated_bytes": 4, "reserved_bytes": 5},
            "must include the captured baseline",
        ),
    ],
)
def test_training_gpu_memory_rejects_inconsistent_cuda_counters(cuda, baseline, message):
    with pytest.raises(RuntimeError, match=message):
        if baseline is None:
            capture_gpu_memory_baseline(cuda)
        else:
            summarize_training_gpu_memory(cuda, baseline)
