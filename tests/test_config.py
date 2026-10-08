import copy
import json
from pathlib import Path

import pytest

from gemma_experiment.config import (
    ConfigValidationError,
    config_sha256,
    load_config,
    validate_config,
)

CONFIG_PATH = Path("configs/experiment.json")


def checked_in_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_checked_in_config_reports_run_budget_and_fingerprint():
    config, report = load_config(CONFIG_PATH)

    assert report["schema_version"] == 1
    assert report["config_sha256"] == config_sha256(config)
    assert len(report["config_sha256"]) == 64
    assert report["derived"] == {
        "effective_train_batch_size": 4,
        "train_examples_consumed": 240,
        "equivalent_training_epochs": 0.1,
        "evaluation_events": 6,
        "checkpoint_events": 6,
    }


def test_fingerprint_is_canonical_and_changes_with_configuration():
    config = checked_in_config()
    reversed_config = dict(reversed(config.items()))
    modified_config = copy.deepcopy(config)
    modified_config["training"]["max_steps"] += 10

    assert config_sha256(config) == config_sha256(reversed_config)
    assert config_sha256(config) != config_sha256(modified_config)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("data", "test_size"), 299, "must equal sample_size"),
        (("training", "per_device_train_batch_size"), True, "must be an integer"),
        (("training", "save_steps"), 15, "divisible by eval_steps"),
        (("evaluation", "generation_examples"), 301, "must not exceed"),
        (("evaluation", "max_new_tokens"), 1024, "must be smaller"),
        (("lora", "dropout"), 1.0, "must be <"),
        (("evaluation", "bootstrap_confidence"), 0.0, "must be >"),
        (("model", "revision"), "main", "full 40-character"),
    ],
)
def test_rejects_invalid_and_incompatible_settings(path, value, message):
    config = checked_in_config()
    section, field = path
    config[section][field] = value

    with pytest.raises(ConfigValidationError, match=message):
        validate_config(config)


def test_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / "experiment.json"
    path.write_text("{not-json}", encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="Invalid JSON"):
        load_config(path)
