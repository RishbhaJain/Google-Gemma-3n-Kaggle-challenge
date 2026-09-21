from pathlib import Path

import pytest

from gemma_experiment.data import (
    deterministic_split,
    normalize_messages,
    read_jsonl,
    sha256_file,
    to_prompt_completion,
    write_jsonl,
)


def rows(count: int = 12):
    return [
        {
            "conversations": [
                {"from": "human", "value": f"question {index}"},
                {"from": "gpt", "value": f"answer {index}"},
            ]
        }
        for index in range(count)
    ]


def test_normalizes_finetome_roles_and_builds_completion():
    example = to_prompt_completion(rows(1)[0])
    assert example["prompt"] == [{"role": "user", "content": "question 0"}]
    assert example["completion"] == [{"role": "assistant", "content": "answer 0"}]


def test_requires_final_assistant_turn():
    with pytest.raises(ValueError, match="end with an assistant"):
        to_prompt_completion(
            {
                "conversations": [
                    {"from": "gpt", "value": "answer"},
                    {"from": "human", "value": "question"},
                ]
            }
        )


def test_rejects_unknown_roles():
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_messages([{"from": "tool", "value": "result"}])


def test_split_is_seeded_disjoint_and_has_expected_sizes():
    first = deterministic_split(rows(), train_size=8, validation_size=2, test_size=2, seed=7)
    second = deterministic_split(rows(), train_size=8, validation_size=2, test_size=2, seed=7)
    assert first == second
    assert {name: len(split) for name, split in first.items()} == {
        "train": 8,
        "validation": 2,
        "test": 2,
    }
    source_sets = [{item["source_index"] for item in split} for split in first.values()]
    assert source_sets[0].isdisjoint(source_sets[1])
    assert source_sets[0].isdisjoint(source_sets[2])
    assert source_sets[1].isdisjoint(source_sets[2])


def test_split_rejects_insufficient_rows():
    with pytest.raises(ValueError, match="Need 13"):
        deterministic_split(rows(), train_size=9, validation_size=2, test_size=2, seed=7)


def test_jsonl_round_trip_and_checksum(tmp_path: Path):
    path = tmp_path / "examples.jsonl"
    examples = [{"id": 1}, {"id": 2}]
    assert write_jsonl(path, examples) == 2
    assert read_jsonl(path) == examples
    assert len(sha256_file(path)) == 64
