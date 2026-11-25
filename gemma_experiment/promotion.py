"""Fail-closed promotion policy for base-versus-QLoRA evaluation results."""

from __future__ import annotations

import math
from typing import Any

PROMOTION_POLICY_SCHEMA_VERSION = 1

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

    return {
        "schema_version": schema_version,
        "result_schema_version": result_schema_version,
        "thresholds": validated_thresholds,
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

    bootstrap = _mapping(result_root.get("paired_bootstrap"), "result.paired_bootstrap")
    thresholds = validated_policy["thresholds"]
    checks = []
    for gate_name, metric_name in _BOOTSTRAP_GATES.items():
        metric = _mapping(bootstrap.get(metric_name), f"result.paired_bootstrap.{metric_name}")
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

    base = _mapping(result_root.get("base"), "result.base")
    candidate = _mapping(result_root.get("qlora"), "result.qlora")
    base_latency = _positive_number(
        base.get("latency_p95_seconds"), "result.base.latency_p95_seconds"
    )
    _positive_number(candidate.get("latency_p95_seconds"), "result.qlora.latency_p95_seconds")
    latency_bootstrap = _mapping(
        bootstrap.get("latency_p95_seconds"),
        "result.paired_bootstrap.latency_p95_seconds",
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
        "eligible": passed == len(checks),
        "passed_checks": passed,
        "failed_checks": len(checks) - passed,
        "checks": checks,
    }
