"""Fail-closed promotion policy for base-versus-QLoRA evaluation results."""

from __future__ import annotations

import math
import re
from typing import Any

PROMOTION_POLICY_SCHEMA_VERSION = 2

_BOOTSTRAP_GATES = {
    "completion_loss_upper_ci": "completion_loss",
    "generation_limit_hit_rate_upper_ci": "generation_limit_hit",
    "empty_response_rate_upper_ci": "empty_response",
    "repeated_4gram_rate_upper_ci": "repeated_4gram_rate",
}
_RATIO_GATES = {
    "peak_gpu_reserved_ratio": "peak_gpu_reserved_gib",
}
_LATENCY_GATE = "latency_p95_upper_ratio"
_REQUIRED_THRESHOLDS = frozenset((*_BOOTSTRAP_GATES, *_RATIO_GATES, _LATENCY_GATE))
_REQUIRED_EVIDENCE = frozenset(
    {
        "experiment_config_sha256",
        "minimum_completion_loss_examples",
        "minimum_generation_examples",
        "minimum_bootstrap_samples",
        "minimum_bootstrap_confidence",
        "require_cuda",
    }
)
_RESULT_SHA256_FIELDS = (
    "dataset_manifest_sha256",
    "test_split_sha256",
    "adapter_config_sha256",
    "adapter_artifact_sha256",
    "training_summary_sha256",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class PromotionValidationError(ValueError):
    """Raised when a result or promotion policy cannot be evaluated safely."""


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PromotionValidationError(f"{field} must be an object")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PromotionValidationError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise PromotionValidationError(f"{field} must be a finite number")
    return number


def _positive_number(value: Any, field: str) -> float:
    number = _finite_number(value, field)
    if number <= 0:
        raise PromotionValidationError(f"{field} must be positive")
    return number


def _positive_integer(value: Any, field: str) -> int:
    if type(value) is not int or value < 1:
        raise PromotionValidationError(f"{field} must be a positive integer")
    return value


def _sha(value: Any, field: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise PromotionValidationError(f"{field} must be a full lowercase hexadecimal digest")
    return value


def _record_list(value: Any, field: str, expected_count: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(record, dict) for record in value):
        raise PromotionValidationError(f"{field} must be a list of record objects")
    if len(value) != expected_count:
        raise PromotionValidationError(
            f"{field} must contain {expected_count} records, found {len(value)}"
        )
    return value


def validate_promotion_policy(policy: dict[str, Any]) -> dict[str, Any]:
    """Validate the complete promotion policy and reject ambiguous thresholds."""
    root = _mapping(policy, "policy")
    schema_version = root.get("schema_version")
    if schema_version != PROMOTION_POLICY_SCHEMA_VERSION:
        raise PromotionValidationError(
            "policy.schema_version must be "
            f"{PROMOTION_POLICY_SCHEMA_VERSION}, got {schema_version!r}"
        )
    result_schema_version = root.get("result_schema_version")
    if type(result_schema_version) is not int or result_schema_version < 1:
        raise PromotionValidationError("policy.result_schema_version must be a positive integer")

    thresholds = _mapping(root.get("thresholds"), "policy.thresholds")
    keys = set(thresholds)
    missing = sorted(_REQUIRED_THRESHOLDS - keys)
    unknown = sorted(keys - _REQUIRED_THRESHOLDS)
    if missing or unknown:
        raise PromotionValidationError(
            f"policy thresholds must match the contract; missing={missing}, unknown={unknown}"
        )
    validated_thresholds = {
        name: _finite_number(value, f"policy.thresholds.{name}")
        for name, value in thresholds.items()
    }
    for name in (*_RATIO_GATES, _LATENCY_GATE):
        if validated_thresholds[name] <= 0:
            raise PromotionValidationError(f"policy.thresholds.{name} must be positive")

    evidence = _mapping(root.get("evidence"), "policy.evidence")
    evidence_keys = set(evidence)
    missing_evidence = sorted(_REQUIRED_EVIDENCE - evidence_keys)
    unknown_evidence = sorted(evidence_keys - _REQUIRED_EVIDENCE)
    if missing_evidence or unknown_evidence:
        raise PromotionValidationError(
            "policy evidence must match the contract; "
            f"missing={missing_evidence}, unknown={unknown_evidence}"
        )
    config_sha256 = _sha(
        evidence["experiment_config_sha256"],
        "policy.evidence.experiment_config_sha256",
        _SHA256,
    )
    minimum_bootstrap_confidence = _finite_number(
        evidence["minimum_bootstrap_confidence"],
        "policy.evidence.minimum_bootstrap_confidence",
    )
    if not 0 < minimum_bootstrap_confidence < 1:
        raise PromotionValidationError(
            "policy.evidence.minimum_bootstrap_confidence must be between zero and one"
        )
    if type(evidence["require_cuda"]) is not bool:
        raise PromotionValidationError("policy.evidence.require_cuda must be a boolean")
    validated_evidence = {
        "experiment_config_sha256": config_sha256,
        "minimum_completion_loss_examples": _positive_integer(
            evidence["minimum_completion_loss_examples"],
            "policy.evidence.minimum_completion_loss_examples",
        ),
        "minimum_generation_examples": _positive_integer(
            evidence["minimum_generation_examples"],
            "policy.evidence.minimum_generation_examples",
        ),
        "minimum_bootstrap_samples": _positive_integer(
            evidence["minimum_bootstrap_samples"],
            "policy.evidence.minimum_bootstrap_samples",
        ),
        "minimum_bootstrap_confidence": minimum_bootstrap_confidence,
        "require_cuda": evidence["require_cuda"],
    }

    return {
        "schema_version": schema_version,
        "result_schema_version": result_schema_version,
        "thresholds": validated_thresholds,
        "evidence": validated_evidence,
    }


def assess_promotion(result: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    """Apply conservative quality, degeneration, latency, and capacity gates."""
    validated_policy = validate_promotion_policy(policy)
    result_root = _mapping(result, "result")
    result_schema_version = result_root.get("schema_version")
    expected_result_schema = validated_policy["result_schema_version"]
    if result_schema_version != expected_result_schema:
        raise PromotionValidationError(
            "result.schema_version does not match the policy: "
            f"expected {expected_result_schema}, got {result_schema_version!r}"
        )

    evidence = validated_policy["evidence"]
    result_config_sha256 = _sha(
        result_root.get("experiment_config_sha256"),
        "result.experiment_config_sha256",
        _SHA256,
    )
    if result_config_sha256 != evidence["experiment_config_sha256"]:
        raise PromotionValidationError(
            "result.experiment_config_sha256 does not match the promotion policy"
        )
    for field in _RESULT_SHA256_FIELDS:
        _sha(result_root.get(field), f"result.{field}", _SHA256)
    _sha(result_root.get("base_model_revision"), "result.base_model_revision", _COMMIT_SHA)

    bootstrap = _mapping(result_root.get("paired_bootstrap"), "result.paired_bootstrap")
    thresholds = validated_policy["thresholds"]
    base = _mapping(result_root.get("base"), "result.base")
    candidate = _mapping(result_root.get("qlora"), "result.qlora")
    completion_loss_examples = _positive_integer(
        base.get("completion_loss_examples"),
        "result.base.completion_loss_examples",
    )
    if candidate.get("completion_loss_examples") != completion_loss_examples:
        raise PromotionValidationError("result variants must use the same completion_loss_examples")
    if completion_loss_examples < evidence["minimum_completion_loss_examples"]:
        raise PromotionValidationError("result has too few completion-loss examples for promotion")
    generation_examples = _positive_integer(
        base.get("generation_examples"),
        "result.base.generation_examples",
    )
    if candidate.get("generation_examples") != generation_examples:
        raise PromotionValidationError("result variants must use the same generation_examples")
    if generation_examples < evidence["minimum_generation_examples"]:
        raise PromotionValidationError("result has too few generation examples for promotion")

    inference_records = _mapping(result_root.get("inference_records"), "result.inference_records")
    generation_records = _mapping(
        inference_records.get("generation"),
        "result.inference_records.generation",
    )
    loss_records = _mapping(
        inference_records.get("completion_loss"),
        "result.inference_records.completion_loss",
    )
    for variant in ("base", "qlora"):
        _record_list(
            generation_records.get(variant),
            f"result.inference_records.generation.{variant}",
            generation_examples,
        )
        _record_list(
            loss_records.get(variant),
            f"result.inference_records.completion_loss.{variant}",
            completion_loss_examples,
        )

    environment = _mapping(result_root.get("environment"), "result.environment")
    if environment.get("cuda_required") is not evidence["require_cuda"]:
        raise PromotionValidationError(
            "result.environment.cuda_required does not match the promotion policy"
        )
    if evidence["require_cuda"]:
        if not isinstance(environment.get("gpu"), str) or not environment["gpu"].strip():
            raise PromotionValidationError("result.environment.gpu must identify the accelerator")
        if not isinstance(environment.get("cuda"), str) or not environment["cuda"].strip():
            raise PromotionValidationError("result.environment.cuda must identify the CUDA version")
        _positive_number(
            environment.get("gpu_total_memory_gib"),
            "result.environment.gpu_total_memory_gib",
        )

    checks = []
    for gate_name, metric_name in _BOOTSTRAP_GATES.items():
        metric = _mapping(bootstrap.get(metric_name), f"result.paired_bootstrap.{metric_name}")
        expected_examples = (
            completion_loss_examples if metric_name == "completion_loss" else generation_examples
        )
        paired_examples = _positive_integer(
            metric.get("paired_examples"),
            f"result.paired_bootstrap.{metric_name}.paired_examples",
        )
        if paired_examples != expected_examples:
            raise PromotionValidationError(
                f"result.paired_bootstrap.{metric_name}.paired_examples "
                f"must equal {expected_examples}"
            )
        bootstrap_samples = _positive_integer(
            metric.get("bootstrap_samples"),
            f"result.paired_bootstrap.{metric_name}.bootstrap_samples",
        )
        if bootstrap_samples < evidence["minimum_bootstrap_samples"]:
            raise PromotionValidationError(
                f"result.paired_bootstrap.{metric_name} has too few bootstrap samples"
            )
        confidence = _finite_number(
            metric.get("confidence"),
            f"result.paired_bootstrap.{metric_name}.confidence",
        )
        if not 0 < confidence < 1:
            raise PromotionValidationError(
                f"result.paired_bootstrap.{metric_name}.confidence must be between zero and one"
            )
        if confidence < evidence["minimum_bootstrap_confidence"]:
            raise PromotionValidationError(
                f"result.paired_bootstrap.{metric_name} has insufficient confidence"
            )
        value = _finite_number(
            metric.get("confidence_interval_high"),
            f"result.paired_bootstrap.{metric_name}.confidence_interval_high",
        )
        threshold = thresholds[gate_name]
        checks.append(
            {
                "name": gate_name,
                "value": value,
                "operator": "<=",
                "threshold": threshold,
                "passed": value <= threshold,
            }
        )

    base_latency = _positive_number(
        base.get("latency_p95_seconds"), "result.base.latency_p95_seconds"
    )
    _positive_number(candidate.get("latency_p95_seconds"), "result.qlora.latency_p95_seconds")
    latency_bootstrap = _mapping(
        bootstrap.get("latency_p95_seconds"),
        "result.paired_bootstrap.latency_p95_seconds",
    )
    latency_examples = _positive_integer(
        latency_bootstrap.get("paired_examples"),
        "result.paired_bootstrap.latency_p95_seconds.paired_examples",
    )
    if latency_examples != generation_examples:
        raise PromotionValidationError(
            "result.paired_bootstrap.latency_p95_seconds.paired_examples "
            f"must equal {generation_examples}"
        )
    latency_samples = _positive_integer(
        latency_bootstrap.get("bootstrap_samples"),
        "result.paired_bootstrap.latency_p95_seconds.bootstrap_samples",
    )
    if latency_samples < evidence["minimum_bootstrap_samples"]:
        raise PromotionValidationError(
            "result.paired_bootstrap.latency_p95_seconds has too few bootstrap samples"
        )
    latency_confidence = _finite_number(
        latency_bootstrap.get("confidence"),
        "result.paired_bootstrap.latency_p95_seconds.confidence",
    )
    if not 0 < latency_confidence < 1:
        raise PromotionValidationError(
            "result.paired_bootstrap.latency_p95_seconds.confidence must be between zero and one"
        )
    if latency_confidence < evidence["minimum_bootstrap_confidence"]:
        raise PromotionValidationError(
            "result.paired_bootstrap.latency_p95_seconds has insufficient confidence"
        )
    latency_delta_high = _finite_number(
        latency_bootstrap.get("confidence_interval_high"),
        "result.paired_bootstrap.latency_p95_seconds.confidence_interval_high",
    )
    latency_upper_ratio = (base_latency + latency_delta_high) / base_latency
    latency_threshold = thresholds[_LATENCY_GATE]
    checks.append(
        {
            "name": _LATENCY_GATE,
            "value": latency_upper_ratio,
            "operator": "<=",
            "threshold": latency_threshold,
            "passed": latency_upper_ratio <= latency_threshold,
        }
    )

    for gate_name, metric_name in _RATIO_GATES.items():
        base_value = _positive_number(base.get(metric_name), f"result.base.{metric_name}")
        candidate_value = _positive_number(
            candidate.get(metric_name), f"result.qlora.{metric_name}"
        )
        value = candidate_value / base_value
        threshold = thresholds[gate_name]
        checks.append(
            {
                "name": gate_name,
                "value": value,
                "operator": "<=",
                "threshold": threshold,
                "passed": value <= threshold,
            }
        )

    passed = sum(check["passed"] for check in checks)
    return {
        "schema_version": PROMOTION_POLICY_SCHEMA_VERSION,
        "result_schema_version": result_schema_version,
        "evidence": {
            "experiment_config_sha256": result_config_sha256,
            "completion_loss_examples": completion_loss_examples,
            "generation_examples": generation_examples,
            "minimum_bootstrap_samples": evidence["minimum_bootstrap_samples"],
            "minimum_bootstrap_confidence": evidence["minimum_bootstrap_confidence"],
            "cuda_required": evidence["require_cuda"],
            "gpu": environment.get("gpu"),
        },
        "eligible": passed == len(checks),
        "passed_checks": passed,
        "failed_checks": len(checks) - passed,
        "checks": checks,
    }
