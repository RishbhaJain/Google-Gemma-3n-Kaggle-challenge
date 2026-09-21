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

from gemma_experiment.data import read_jsonl, sha256_file
from gemma_experiment.metrics import exact_match, perplexity, summarize_generations, token_f1


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
) -> float:
    import torch

    total_negative_log_likelihood = 0.0
    total_tokens = 0
    for example in examples:
        prompt, full_text, _ = render_example(tokenizer, example)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        encoded = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        labels = input_ids.clone()
        prompt_length = min(len(prompt_ids), labels.shape[1])
        labels[:, :prompt_length] = -100
        supervised_tokens = int((labels[:, 1:] != -100).sum().item())
        if supervised_tokens == 0:
            continue
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_negative_log_likelihood += float(output.loss.item()) * supervised_tokens
        total_tokens += supervised_tokens
    if total_tokens == 0:
        raise RuntimeError("No completion tokens remained after tokenization")
    return total_negative_log_likelihood / total_tokens


def evaluate_variant(
    *,
    model_path: str,
    examples: list[dict[str, Any]],
    max_length: int,
    generation_examples: int,
    max_new_tokens: int,
    load_in_4bit: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from unsloth import FastModel

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_length,
        load_in_4bit=load_in_4bit,
    )
    FastModel.for_inference(model)

    mean_loss = completion_loss(model, tokenizer, examples, max_length)
    records = []
    for example in examples[:generation_examples]:
        prompt, _, reference = render_example(tokenizer, example)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - started
        generated_ids = output[0, encoded["input_ids"].shape[1] :]
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
            "completion_loss": mean_loss,
            "completion_perplexity": perplexity(mean_loss),
            "test_examples": len(examples),
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
    return metrics, environment


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_config = config["model"]
    data_config = config["data"]
    train_config = config["training"]
    eval_config = config["evaluation"]

    test_path = Path(data_config["output_dir"]) / "test.jsonl"
    manifest_path = Path(data_config["output_dir"]) / "manifest.json"
    adapter_path = Path(train_config["adapter_dir"])
    for path in (test_path, manifest_path, adapter_path):
        if not path.exists():
            raise FileNotFoundError(f"Required experiment artifact does not exist: {path}")
    examples = read_jsonl(test_path)

    common = {
        "examples": examples,
        "max_length": model_config["max_sequence_length"],
        "generation_examples": min(eval_config["generation_examples"], len(examples)),
        "max_new_tokens": eval_config["max_new_tokens"],
        "load_in_4bit": model_config["load_in_4bit"],
    }
    base_metrics, environment = evaluate_variant(model_path=model_config["name"], **common)
    adapter_metrics, _ = evaluate_variant(model_path=str(adapter_path), **common)

    result = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "test_split_sha256": sha256_file(test_path),
        "base_model": model_config["name"],
        "adapter": str(adapter_path),
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
        "environment": environment,
        "config": config,
        "metric_notes": {
            "completion_loss": "Token-weighted NLL over assistant completion tokens only.",
            "exact_match_and_f1": (
                "Reference-overlap diagnostics for open-ended data; interpret with "
                "completion loss and qualitative review."
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
