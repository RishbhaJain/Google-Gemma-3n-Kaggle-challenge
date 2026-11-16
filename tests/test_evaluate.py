import pytest

from scripts.evaluate import (
    DELTA_METRICS,
    RESULT_SCHEMA_VERSION,
    build_comparison_report,
    build_loss_slice_report,
    generation_limit_hit,
    select_generation_examples,
    summarize_peak_gpu_memory,
    validate_benchmark_runtime,
    validate_paired_generation_records,
    validate_paired_loss_records,
)


def comparison_metrics(value):
    return {metric: value for metric in DELTA_METRICS}


def generation_record(index, *, candidate=False):
    return {
        "example_index": float(index),
        "prompt_tokens": float(64 + index),
        "generated_tokens": 8.0,
        "exact_match": float(candidate),
        "token_f1": 0.9 if candidate else 0.7,
        "latency_seconds": 0.8 if candidate else 1.0,
        "generation_limit_hit": 0.0,
        "empty_response": 0.0,
        "unique_token_ratio": 0.8 if candidate else 0.6,
        "repeated_4gram_rate": 0.1 if candidate else 0.2,
    }


def loss_record(index, loss, tokens):
    return {
        "example_index": index,
        "mean_loss": loss,
        "scored_tokens": tokens,
        "original_prompt_tokens": 64 + index,
        "completion_tokens": tokens,
    }


def test_generation_limit_hit_requires_full_budget_without_eos():
    assert generation_limit_hit([10, 11], max_new_tokens=3, eos_token_id=2) == 0.0
    assert generation_limit_hit([10, 11, 2], max_new_tokens=3, eos_token_id=2) == 0.0
    assert generation_limit_hit([10, 11, 12], max_new_tokens=3, eos_token_id=2) == 1.0
    assert generation_limit_hit([10, 11, 12], max_new_tokens=3, eos_token_id=None) == 1.0

    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        generation_limit_hit([], max_new_tokens=0, eos_token_id=2)


class FakeCuda:
    def __init__(self, *, available, allocated=0, reserved=0):
        self.available = available
        self.allocated = allocated
        self.reserved = reserved

    def is_available(self):
        return self.available

    def max_memory_allocated(self):
        return self.allocated

    def max_memory_reserved(self):
        return self.reserved


def test_gpu_benchmark_fails_closed_without_cuda():
    with pytest.raises(RuntimeError, match="refusing to publish CPU"):
        validate_benchmark_runtime(cuda_available=False, require_cuda=True)


@pytest.mark.parametrize("cuda_available", [False, True])
def test_runtime_contract_allows_explicit_non_gpu_runs(cuda_available):
    validate_benchmark_runtime(cuda_available=cuda_available, require_cuda=False)


def test_peak_gpu_memory_separates_allocated_and_reserved_capacity():
    gibibyte = 1024**3

    summary = summarize_peak_gpu_memory(
        FakeCuda(available=True, allocated=2 * gibibyte, reserved=3 * gibibyte)
    )

    assert summary == {
        "peak_gpu_allocated_gib": 2.0,
        "peak_gpu_reserved_gib": 3.0,
        "peak_gpu_allocator_overhead_gib": 1.0,
    }


def test_peak_gpu_memory_reports_zero_without_cuda():
    assert summarize_peak_gpu_memory(FakeCuda(available=False)) == {
        "peak_gpu_allocated_gib": 0.0,
        "peak_gpu_reserved_gib": 0.0,
        "peak_gpu_allocator_overhead_gib": 0.0,
    }


def test_peak_gpu_memory_rejects_inconsistent_cuda_counters():
    with pytest.raises(RuntimeError, match="reserved bytes must be at least allocated"):
        summarize_peak_gpu_memory(FakeCuda(available=True, allocated=2, reserved=1))


def test_generation_warmups_are_disjoint_from_measured_examples():
    examples = [{"source_index": index} for index in range(7)]

    warmups, measured = select_generation_examples(examples, measured_count=4, warmup_count=2)

    assert [item["source_index"] for item in warmups] == [4, 5]
    assert [item["source_index"] for item in measured] == [0, 1, 2, 3]
    assert not {id(item) for item in warmups} & {id(item) for item in measured}


def test_generation_rejects_insufficient_disjoint_examples():
    with pytest.raises(ValueError, match="too few examples"):
        select_generation_examples(
            [{"source_index": index} for index in range(4)],
            measured_count=3,
            warmup_count=2,
        )


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


def test_paired_generation_records_require_identical_inputs():
    records = [
        {
            "example_index": 0.0,
            "latency_seconds": 1.0,
            "prompt_tokens": 64.0,
            "generated_tokens": 8.0,
        },
        {
            "example_index": 1.0,
            "latency_seconds": 2.0,
            "prompt_tokens": 128.0,
            "generated_tokens": 12.0,
        },
    ]

    validate_paired_generation_records(records, [dict(record) for record in records])

    with pytest.raises(ValueError, match="indices are not aligned"):
        validate_paired_generation_records(
            records,
            [{**records[0], "example_index": 1.0}, records[1]],
        )

    with pytest.raises(ValueError, match="prompt_tokens differs"):
        validate_paired_generation_records(
            records,
            [records[0], {**records[1], "prompt_tokens": 127.0}],
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


def test_comparison_report_contract_is_versioned_paired_and_directional():
    base_records = [generation_record(0), generation_record(1)]
    adapter_records = [
        generation_record(0, candidate=True),
        generation_record(1, candidate=True),
    ]
    base_loss_records = [loss_record(0, 1.0, 9), loss_record(1, 3.0, 1)]
    adapter_loss_records = [loss_record(0, 0.5, 9), loss_record(1, 4.0, 1)]

    report = build_comparison_report(
        comparison_metrics(1.0),
        comparison_metrics(1.25),
        base_records,
        adapter_records,
        base_loss_records,
        adapter_loss_records,
        bootstrap_options={"confidence": 0.95, "samples": 200, "seed": 7},
    )

    assert report["schema_version"] == RESULT_SCHEMA_VERSION
    assert set(report["delta"]) == set(DELTA_METRICS)
    assert set(report["delta"].values()) == {0.25}
    bootstrap = report["paired_bootstrap"]
    assert bootstrap["completion_loss"]["observed_delta"] == pytest.approx(-0.35)
    assert bootstrap["completion_loss"]["paired_tokens"] == 10
    assert bootstrap["completion_loss"]["improvement_direction"] == "lower"
    assert bootstrap["latency_seconds"]["improvement_direction"] == "lower"
    assert bootstrap["unique_token_ratio"]["improvement_direction"] == "higher"
    assert report["completion_loss_slices"]["completion_tokens"]["has_observed_regression"] is False


def test_comparison_report_rejects_unpaired_records():
    base_records = [generation_record(0)]
    adapter_records = [{**generation_record(0, candidate=True), "example_index": 1.0}]
    loss_records = [loss_record(0, 1.0, 8)]

    with pytest.raises(ValueError, match="indices are not aligned"):
        build_comparison_report(
            comparison_metrics(1.0),
            comparison_metrics(1.25),
            base_records,
            adapter_records,
            loss_records,
            loss_records,
            bootstrap_options={"confidence": 0.95, "samples": 20, "seed": 7},
        )
