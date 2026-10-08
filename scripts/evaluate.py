#!/usr/bin/env python3
"""Compare the base model and QLoRA adapter on the untouched test split."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma_experiment.config import load_config
from gemma_experiment.data import read_jsonl, sha256_file
from gemma_experiment.integrity import (
    build_completion_encoding,
    validate_adapter_base_revision,
    validate_model_revision,
    validate_split_checksum,
)
from gemma_experiment.metrics import (
    exact_match,
    paired_bootstrap_delta,
    perplexity,
    summarize_generations,
    token_f1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    return parser.parse_args()


def render_example(tokenizer: Any, example: dict[str, Any]) -> tuple[str, str, str]:
    prompt = tokenizer.apply_chat_template(
        example["prompt"], tokenize=False, add_generation_prompt=True
    )
    reference = example["completion"][0]["content"]
    full_text = tokenizer.apply_chat_template(
        example["prompt"] + example["completion"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return prompt, full_text, reference


def completion_loss(
    model: Any, tokenizer: Any, examples: list[dict[str, Any]], max_length: int
) -> dict[str, float | int]:
    import torch

    total_negative_log_likelihood = 0.0
    total_tokens = 0
    total_removed_prompt_tokens = 0
    scored_examples = 0
    for example in examples:
        prompt, full_text, _ = render_example(tokenizer, example)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        encoded = build_completion_encoding(
            prompt_ids=prompt_ids,
            full_ids=full_ids,
            max_length=max_length,
        )
        input_ids = torch.tensor([encoded.input_ids], dtype=torch.long, device=model.device)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor([encoded.labels], dtype=torch.long, device=model.device)
        supervised_tokens = int((labels[:, 1:] != -100).sum().item())
        if supervised_tokens != encoded.completion_tokens:
            raise RuntimeError(
                "Completion-token accounting mismatch: "
                f"expected {encoded.completion_tokens}, scored {supervised_tokens}"
            )
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_negative_log_likelihood += float(output.loss.item()) * supervised_tokens
        total_tokens += supervised_tokens
        total_removed_prompt_tokens += encoded.removed_prompt_tokens
        scored_examples += 1
    if total_tokens == 0:
        raise RuntimeError("No completion tokens were scored")
    return {
        "mean_loss": total_negative_log_likelihood / total_tokens,
        "scored_examples": scored_examples,
        "scored_tokens": total_tokens,
        "removed_prompt_tokens": total_removed_prompt_tokens,
    }


def evaluate_variant(
    *,
    model_path: str,
    examples: list[dict[str, Any]],
    max_length: int,
    generation_examples: int,
    max_new_tokens: int,
    load_in_4bit: bool,
    revision: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, float]]]:
    import torch
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    load_options = {
        "model_name": model_path,
        "max_seq_length": max_length,
        "load_in_4bit": load_in_4bit,
    }
    if revision is not None:
        load_options["revision"] = revision
    model, tokenizer = FastModel.from_pretrained(**load_options)
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
    FastModel.for_inference(model)

    loss_summary = completion_loss(model, tokenizer, examples, max_length)
    records = []
    for example in examples[:generation_examples]:
        prompt, _, reference = render_example(tokenizer, example)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_budget = max_length - max_new_tokens
        if prompt_budget < 1:
            raise ValueError("max_new_tokens must be smaller than max_length")
        prompt_ids = prompt_ids[-prompt_budget:]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
        attention_mask = torch.ones_like(input_ids)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - started
        generated_ids = output[0, input_ids.shape[1] :]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        records.append(
            {
                "exact_match": exact_match(prediction, reference),
                "token_f1": token_f1(prediction, reference),
                "latency_seconds": latency,
                "generated_tokens": float(len(generated_ids)),
            }
        )

    metrics = summarize_generations(records)
    metrics.update(
        {
            "completion_loss": loss_summary["mean_loss"],
            "completion_perplexity": perplexity(loss_summary["mean_loss"]),
            "test_examples": len(examples),
            "completion_loss_examples": loss_summary["scored_examples"],
            "completion_loss_tokens": loss_summary["scored_tokens"],
            "left_truncated_prompt_tokens": loss_summary["removed_prompt_tokens"],
            "generation_examples": len(records),
            "peak_gpu_memory_gib": (
                torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
            ),
        }
    )
    environment = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, environment, records


def main() -> None:
    args = parse_args()
    config, config_report = load_config(args.config)
    model_config = config["model"]
    data_config = config["data"]
    train_config = config["training"]
    eval_config = config["evaluation"]
    model_revision = validate_model_revision(model_config["revision"])

    test_path = Path(data_config["output_dir"]) / "test.jsonl"
    manifest_path = Path(data_config["output_dir"]) / "manifest.json"
    adapter_path = Path(train_config["adapter_dir"])
    for path in (test_path, manifest_path, adapter_path):
        if not path.exists():
            raise FileNotFoundError(f"Required experiment artifact does not exist: {path}")
    test_split_sha256 = validate_split_checksum(test_path, manifest_path, "test")
    adapter_config_sha256 = validate_adapter_base_revision(
        adapter_path, model_config["name"], model_revision
    )
    examples = read_jsonl(test_path)

    common = {
        "examples": examples,
        "max_length": model_config["max_sequence_length"],
        "generation_examples": min(eval_config["generation_examples"], len(examples)),
        "max_new_tokens": eval_config["max_new_tokens"],
        "load_in_4bit": model_config["load_in_4bit"],
    }
    base_metrics, environment, base_records = evaluate_variant(
        model_path=model_config["name"], revision=model_revision, **common
    )
    adapter_metrics, _, adapter_records = evaluate_variant(model_path=str(adapter_path), **common)
    bootstrap_options = {
        "confidence": eval_config["bootstrap_confidence"],
        "samples": eval_config["bootstrap_samples"],
        "seed": eval_config["bootstrap_seed"],
    }

    result = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "experiment_config_sha256": config_report["config_sha256"],
        "run_budget": config_report["derived"],
        "test_split_sha256": test_split_sha256,
        "base_model": model_config["name"],
        "base_model_revision": model_revision,
        "adapter": str(adapter_path),
        "adapter_config_sha256": adapter_config_sha256,
        "base": base_metrics,
        "qlora": adapter_metrics,
        "delta": {
            key: adapter_metrics[key] - base_metrics[key]
            for key in (
                "completion_loss",
                "completion_perplexity",
                "exact_match",
                "token_f1",
                "latency_p50_seconds",
                "latency_p95_seconds",
                "tokens_per_second",
                "peak_gpu_memory_gib",
            )
        },
        "paired_bootstrap": {
            metric: paired_bootstrap_delta(
                [record[metric] for record in base_records],
                [record[metric] for record in adapter_records],
                **bootstrap_options,
            )
            for metric in ("exact_match", "token_f1")
        },
        "environment": environment,
        "config": config,
        "metric_notes": {
            "completion_loss": (
                "Token-weighted NLL over every assistant completion token. Long prompts are "
                "left-truncated; completions are never silently truncated."
            ),
            "exact_match_and_f1": (
                "Reference-overlap diagnostics for open-ended data; interpret with "
                "completion loss and qualitative review. Paired bootstrap intervals "
                "resample prompts, preserving base/adapter pairing."
            ),
            "latency": "Greedy generation, batch size one, measured on a single host.",
        },
    }
    result_path = Path(eval_config["results_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
