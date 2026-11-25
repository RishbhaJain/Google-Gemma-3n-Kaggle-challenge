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
        "schema_version": 1,
        "result_schema_version": 2,
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
    return {
        "schema_version": 2,
        "paired_bootstrap": {
            "completion_loss": {"confidence_interval_high": -0.01},
            "generation_limit_hit": {"confidence_interval_high": 0.03},
            "empty_response": {"confidence_interval_high": 0.01},
            "repeated_4gram_rate": {"confidence_interval_high": 0.04},
            "latency_p95_seconds": {"confidence_interval_high": 0.15},
        },
        "base": {
            "latency_p95_seconds": 1.0,
            "peak_gpu_reserved_gib": 10.0,
        },
        "qlora": {
            "latency_p95_seconds": 1.1,
            "peak_gpu_reserved_gib": 11.0,
        },
    }


def test_promotion_passes_only_when_every_guardrail_passes():
    report = assess_promotion(result(), policy())

    assert report["eligible"] is True
    assert report["passed_checks"] == 6
    assert report["failed_checks"] == 0
    assert {check["name"] for check in report["checks"]} == set(policy()["thresholds"])


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
