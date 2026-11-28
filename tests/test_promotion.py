import json
import sys
from pathlib import Path

import pytest

from gemma_experiment.promotion import (
    PromotionValidationError,
    assess_promotion,
    validate_promotion_policy,
)
from scripts.check_promotion import main


def policy():
    return {
        "schema_version": 2,
        "result_schema_version": 2,
        "evidence": {
            "experiment_config_sha256": "a" * 64,
            "minimum_completion_loss_examples": 2,
            "minimum_generation_examples": 2,
            "minimum_bootstrap_samples": 2_000,
            "minimum_bootstrap_confidence": 0.95,
            "require_cuda": True,
        },
        "thresholds": {
            "completion_loss_upper_ci": 0.0,
            "generation_limit_hit_rate_upper_ci": 0.05,
            "empty_response_rate_upper_ci": 0.02,
            "repeated_4gram_rate_upper_ci": 0.05,
            "latency_p95_upper_ratio": 1.2,
            "peak_gpu_reserved_ratio": 1.2,
        },
    }


def result():
    bootstrap_evidence = {
        "paired_examples": 2,
        "bootstrap_samples": 2_000,
        "confidence": 0.95,
    }
    return {
        "schema_version": 2,
        "experiment_config_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "test_split_sha256": "c" * 64,
        "adapter_config_sha256": "d" * 64,
        "adapter_artifact_sha256": "e" * 64,
        "training_summary_sha256": "f" * 64,
        "base_model_revision": "1" * 40,
        "paired_bootstrap": {
            "completion_loss": {
                "confidence_interval_high": -0.01,
                **bootstrap_evidence,
            },
            "generation_limit_hit": {
                "confidence_interval_high": 0.03,
                **bootstrap_evidence,
            },
            "empty_response": {
                "confidence_interval_high": 0.01,
                **bootstrap_evidence,
            },
            "repeated_4gram_rate": {
                "confidence_interval_high": 0.04,
                **bootstrap_evidence,
            },
            "latency_p95_seconds": {
                "confidence_interval_high": 0.15,
                **bootstrap_evidence,
            },
        },
        "base": {
            "latency_p95_seconds": 1.0,
            "peak_gpu_reserved_gib": 10.0,
            "completion_loss_examples": 2,
            "generation_examples": 2,
        },
        "qlora": {
            "latency_p95_seconds": 1.1,
            "peak_gpu_reserved_gib": 11.0,
            "completion_loss_examples": 2,
            "generation_examples": 2,
        },
        "inference_records": {
            "generation": {
                "base": [{}, {}],
                "qlora": [{}, {}],
            },
            "completion_loss": {
                "base": [{}, {}],
                "qlora": [{}, {}],
            },
        },
        "environment": {
            "cuda_required": True,
            "gpu": "NVIDIA Test GPU",
            "cuda": "12.4",
            "gpu_total_memory_gib": 24.0,
        },
    }


def test_promotion_passes_only_when_every_guardrail_passes():
    report = assess_promotion(result(), policy())

    assert report["eligible"] is True
    assert report["passed_checks"] == 6
    assert report["failed_checks"] == 0
    assert {check["name"] for check in report["checks"]} == set(policy()["thresholds"])
    assert report["evidence"]["generation_examples"] == 2
    assert report["evidence"]["gpu"] == "NVIDIA Test GPU"


def test_promotion_reports_each_failed_guardrail():
    failed_result = result()
    failed_result["paired_bootstrap"]["completion_loss"]["confidence_interval_high"] = 0.02
    failed_result["paired_bootstrap"]["latency_p95_seconds"]["confidence_interval_high"] = 0.3

    report = assess_promotion(failed_result, policy())

    assert report["eligible"] is False
    assert report["failed_checks"] == 2
    assert {check["name"] for check in report["checks"] if not check["passed"]} == {
        "completion_loss_upper_ci",
        "latency_p95_upper_ratio",
    }


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda value: value.update(schema_version=3), "result.schema_version"),
        (
            lambda value: value["base"].update(latency_p95_seconds=0.0),
            "result.base.latency_p95_seconds must be positive",
        ),
        (
            lambda value: value["paired_bootstrap"]["empty_response"].update(
                confidence_interval_high=float("nan")
            ),
            "must be a finite number",
        ),
    ],
)
def test_promotion_fails_closed_on_invalid_results(mutation, message):
    invalid_result = result()
    mutation(invalid_result)

    with pytest.raises(PromotionValidationError, match=message):
        assess_promotion(invalid_result, policy())


def test_policy_rejects_missing_and_unknown_thresholds():
    invalid_policy = policy()
    del invalid_policy["thresholds"]["empty_response_rate_upper_ci"]
    invalid_policy["thresholds"]["unreviewed_metric"] = 1.0

    with pytest.raises(PromotionValidationError, match="missing=.*empty_response.*unknown"):
        validate_promotion_policy(invalid_policy)


@pytest.mark.parametrize(
    "mutation,message",
    [
        (
            lambda value: value.update(experiment_config_sha256="0" * 64),
            "does not match",
        ),
        (
            lambda value: value["base"].update(generation_examples=1),
            "same generation_examples",
        ),
        (
            lambda value: value["paired_bootstrap"]["completion_loss"].update(
                bootstrap_samples=100
            ),
            "too few bootstrap samples",
        ),
        (
            lambda value: value["paired_bootstrap"]["completion_loss"].update(confidence=1.5),
            "must be between zero and one",
        ),
        (
            lambda value: value["environment"].update(gpu=None),
            "must identify the accelerator",
        ),
        (
            lambda value: value["inference_records"]["generation"]["qlora"].pop(),
            "must contain 2 records",
        ),
    ],
)
def test_promotion_rejects_incomplete_or_stale_evidence(mutation, message):
    invalid_result = result()
    mutation(invalid_result)

    with pytest.raises(PromotionValidationError, match=message):
        assess_promotion(invalid_result, policy())


def test_checked_in_policy_matches_the_versioned_contract():
    checked_in = json.loads(Path("configs/promotion_policy.json").read_text(encoding="utf-8"))

    assert validate_promotion_policy(checked_in) == checked_in


def test_cli_returns_nonzero_for_an_ineligible_adapter(tmp_path, monkeypatch, capsys):
    result_path = tmp_path / "result.json"
    policy_path = tmp_path / "policy.json"
    failed_result = result()
    failed_result["qlora"]["peak_gpu_reserved_gib"] = 13.0
    result_path.write_text(json.dumps(failed_result), encoding="utf-8")
    policy_path.write_text(json.dumps(policy()), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_promotion",
            "--result",
            str(result_path),
            "--policy",
            str(policy_path),
        ],
    )

    assert main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report["eligible"] is False
    assert report["failed_checks"] == 1
