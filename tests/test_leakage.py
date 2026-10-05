import json

import pytest

from gemma_experiment.leakage import audit_split_leakage, normalize_prompt, require_leakage_free


def example(text: str, source_index: int) -> dict:
    return {
        "prompt": [{"role": "user", "content": text}],
        "completion": [{"role": "assistant", "content": "response"}],
        "source_index": source_index,
    }


def test_normalization_is_unicode_case_and_punctuation_stable():
    first = normalize_prompt(example("Explain CAFÉ pricing!", 1))
    second = normalize_prompt(example("explain café pricing", 2))
    assert first == second


def test_audit_detects_normalized_exact_duplicate_across_splits():
    report = audit_split_leakage(
        {
            "train": [example("How does QLoRA work?", 10)],
            "test": [example("HOW does qlora work", 20)],
        }
    )
    assert report["summary"] == {
        "exact_cross_split_pairs": 1,
        "near_cross_split_pairs": 0,
        "passed": False,
    }
    assert report["exact_findings"][0]["left_source_index"] == 20
    assert report["exact_findings"][0]["right_source_index"] == 10


def test_audit_detects_near_duplicate_with_jaccard_score():
    report = audit_split_leakage(
        {
            "train": [example("explain low rank adapter training for language models", 1)],
            "validation": [example("explain low rank adapter tuning for language models", 2)],
        },
        near_duplicate_threshold=0.5,
        shingle_size=2,
    )
    assert report["summary"]["near_cross_split_pairs"] == 1
    assert 0.5 <= report["near_findings"][0]["similarity"] < 1


def test_same_split_duplicates_do_not_count_as_evaluation_leakage():
    report = audit_split_leakage(
        {
            "train": [example("same prompt", 1), example("same prompt", 2)],
            "test": [example("unrelated held out question", 3)],
        },
        near_duplicate_threshold=0.8,
    )
    assert report["summary"]["passed"] is True


def test_report_never_contains_dataset_text():
    secret = "private participant medical detail"
    report = audit_split_leakage(
        {
            "train": [example(secret, 1)],
            "test": [example(secret.upper(), 2)],
        }
    )
    assert secret not in json.dumps(report).casefold()


def test_require_leakage_free_rejects_failed_audit():
    report = audit_split_leakage(
        {"train": [example("duplicate", 1)], "test": [example("duplicate", 2)]}
    )
    with pytest.raises(ValueError, match="1 exact"):
        require_leakage_free(report)


def test_clean_splits_pass_and_are_deterministic():
    splits = {
        "train": [example("teach me matrix multiplication", 1)],
        "validation": [example("summarize a poem about rain", 2)],
        "test": [example("write a database migration plan", 3)],
    }
    first = audit_split_leakage(splits)
    second = audit_split_leakage(splits)
    assert first == second
    require_leakage_free(first)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"near_duplicate_threshold": 0}, "threshold"),
        ({"shingle_size": 0}, "shingle_size"),
        ({"max_findings": 0}, "max_findings"),
    ],
)
def test_audit_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        audit_split_leakage(
            {"train": [example("one", 1)], "test": [example("two", 2)]},
            **kwargs,
        )
