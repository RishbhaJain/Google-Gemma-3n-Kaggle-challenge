"""Fail-fast validation and provenance for experiment configuration."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = 1
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class ConfigValidationError(ValueError):
    """Raised when an experiment configuration violates the run contract."""


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigValidationError(f"{field} must be an object")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"{field} must be a non-empty string")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigValidationError(f"{field} must be a boolean")
    return value


def _integer(value: Any, field: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigValidationError(f"{field} must be an integer >= {minimum}")
    return value


def _number(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigValidationError(f"{field} must be a number")
    number = float(value)
    if minimum is not None:
        invalid = number < minimum if minimum_inclusive else number <= minimum
        if invalid:
            comparison = ">=" if minimum_inclusive else ">"
            raise ConfigValidationError(f"{field} must be {comparison} {minimum}")
    if maximum is not None:
        invalid = number > maximum if maximum_inclusive else number >= maximum
        if invalid:
            comparison = "<=" if maximum_inclusive else "<"
            raise ConfigValidationError(f"{field} must be {comparison} {maximum}")
    return number


def _seed(value: Any, field: str) -> int:
    seed = _integer(value, field, minimum=0)
    if seed > 2**32 - 1:
        raise ConfigValidationError(f"{field} must be <= {2**32 - 1}")
    return seed


def _revision(value: Any, field: str) -> str:
    revision = _string(value, field)
    if not _COMMIT_SHA.fullmatch(revision):
        raise ConfigValidationError(f"{field} must be a full 40-character lowercase commit SHA")
    return revision


def config_sha256(config: dict[str, Any]) -> str:
    """Hash the canonical JSON representation of an experiment configuration."""

    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate cross-field invariants and return immutable run metadata."""

    root = _mapping(config, "config")
    schema_version = _integer(root.get("schema_version"), "schema_version")
    if schema_version != CONFIG_SCHEMA_VERSION:
        raise ConfigValidationError(
            f"schema_version must be {CONFIG_SCHEMA_VERSION}, got {schema_version}"
        )

    model = _mapping(root.get("model"), "model")
    _string(model.get("name"), "model.name")
    _revision(model.get("revision"), "model.revision")
    _boolean(model.get("load_in_4bit"), "model.load_in_4bit")
    max_sequence_length = _integer(
        model.get("max_sequence_length"), "model.max_sequence_length", minimum=2
    )

    data = _mapping(root.get("data"), "data")
    _string(data.get("dataset"), "data.dataset")
    _revision(data.get("revision"), "data.revision")
    _string(data.get("source_split"), "data.source_split")
    sample_size = _integer(data.get("sample_size"), "data.sample_size")
    train_size = _integer(data.get("train_size"), "data.train_size")
    validation_size = _integer(data.get("validation_size"), "data.validation_size")
    test_size = _integer(data.get("test_size"), "data.test_size")
    if train_size + validation_size + test_size != sample_size:
        raise ConfigValidationError(
            "data train_size + validation_size + test_size must equal sample_size"
        )
    _seed(data.get("seed"), "data.seed")
    _string(data.get("output_dir"), "data.output_dir")
    leakage = _mapping(data.get("leakage_audit"), "data.leakage_audit")
    _number(
        leakage.get("near_duplicate_threshold"),
        "data.leakage_audit.near_duplicate_threshold",
        minimum=0,
        maximum=1,
        minimum_inclusive=False,
    )
    _integer(leakage.get("shingle_size"), "data.leakage_audit.shingle_size")
    _integer(leakage.get("max_findings"), "data.leakage_audit.max_findings")

    lora = _mapping(root.get("lora"), "lora")
    _integer(lora.get("rank"), "lora.rank")
    _integer(lora.get("alpha"), "lora.alpha")
    _number(lora.get("dropout"), "lora.dropout", minimum=0, maximum=1, maximum_inclusive=False)
    _string(lora.get("bias"), "lora.bias")
    tuning_flags = [
        _boolean(lora.get(field), f"lora.{field}")
        for field in (
            "finetune_vision_layers",
            "finetune_language_layers",
            "finetune_attention_modules",
            "finetune_mlp_modules",
        )
    ]
    if not any(tuning_flags):
        raise ConfigValidationError("lora must enable at least one finetune target")

    training = _mapping(root.get("training"), "training")
    output_dir = _string(training.get("output_dir"), "training.output_dir")
    adapter_dir = _string(training.get("adapter_dir"), "training.adapter_dir")
    if Path(output_dir) == Path(adapter_dir):
        raise ConfigValidationError("training.output_dir and training.adapter_dir must differ")
    train_batch_size = _integer(
        training.get("per_device_train_batch_size"),
        "training.per_device_train_batch_size",
    )
    _integer(
        training.get("per_device_eval_batch_size"),
        "training.per_device_eval_batch_size",
    )
    accumulation_steps = _integer(
        training.get("gradient_accumulation_steps"),
        "training.gradient_accumulation_steps",
    )
    _number(
        training.get("learning_rate"),
        "training.learning_rate",
        minimum=0,
        minimum_inclusive=False,
    )
    _number(training.get("weight_decay"), "training.weight_decay", minimum=0)
    _integer(training.get("warmup_steps"), "training.warmup_steps", minimum=0)
    max_steps = _integer(training.get("max_steps"), "training.max_steps")
    eval_steps = _integer(training.get("eval_steps"), "training.eval_steps")
    save_steps = _integer(training.get("save_steps"), "training.save_steps")
    _integer(training.get("logging_steps"), "training.logging_steps")
    _string(training.get("optimizer"), "training.optimizer")
    _string(training.get("scheduler"), "training.scheduler")
    _seed(training.get("seed"), "training.seed")
    if max_steps < eval_steps or max_steps < save_steps:
        raise ConfigValidationError("training.max_steps must be >= both eval_steps and save_steps")
    if save_steps % eval_steps:
        raise ConfigValidationError(
            "training.save_steps must be divisible by eval_steps when loading the best model"
        )

    evaluation = _mapping(root.get("evaluation"), "evaluation")
    generation_examples = _integer(
        evaluation.get("generation_examples"), "evaluation.generation_examples"
    )
    if generation_examples > test_size:
        raise ConfigValidationError("evaluation.generation_examples must not exceed data.test_size")
    max_new_tokens = _integer(evaluation.get("max_new_tokens"), "evaluation.max_new_tokens")
    if max_new_tokens >= max_sequence_length:
        raise ConfigValidationError(
            "evaluation.max_new_tokens must be smaller than model.max_sequence_length"
        )
    _integer(evaluation.get("bootstrap_samples"), "evaluation.bootstrap_samples")
    _number(
        evaluation.get("bootstrap_confidence"),
        "evaluation.bootstrap_confidence",
        minimum=0,
        maximum=1,
        minimum_inclusive=False,
        maximum_inclusive=False,
    )
    _seed(evaluation.get("bootstrap_seed"), "evaluation.bootstrap_seed")
    _string(evaluation.get("results_path"), "evaluation.results_path")

    effective_batch_size = train_batch_size * accumulation_steps
    examples_consumed = max_steps * effective_batch_size
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "config_sha256": config_sha256(root),
        "derived": {
            "effective_train_batch_size": effective_batch_size,
            "train_examples_consumed": examples_consumed,
            "equivalent_training_epochs": examples_consumed / train_size,
            "evaluation_events": max_steps // eval_steps,
            "checkpoint_events": max_steps // save_steps,
        },
    }


def load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a JSON config, validate it, and return the config plus run metadata."""

    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigValidationError(f"Invalid JSON in {path}: {exc}") from exc
    return config, validate_config(config)
