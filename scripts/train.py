#!/usr/bin/env python3
"""Run the configured Gemma 3n QLoRA experiment with held-out validation."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

from gemma_experiment.integrity import validate_split_checksum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_config = config["model"]
    data_config = config["data"]
    lora_config = config["lora"]
    train_config = config["training"]

    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    data_dir = Path(data_config["output_dir"])
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing data manifest {manifest_path}; run prepare_data.py first")
    data_files = {split: str(data_dir / f"{split}.jsonl") for split in ("train", "validation")}
    for split, path in data_files.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing prepared split {path}; run prepare_data.py first")
        validate_split_checksum(Path(path), manifest_path, split)
    dataset = load_dataset("json", data_files=data_files)

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_config["name"],
        max_seq_length=model_config["max_sequence_length"],
        load_in_4bit=model_config["load_in_4bit"],
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=lora_config["finetune_vision_layers"],
        finetune_language_layers=lora_config["finetune_language_layers"],
        finetune_attention_modules=lora_config["finetune_attention_modules"],
        finetune_mlp_modules=lora_config["finetune_mlp_modules"],
        r=lora_config["rank"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        bias=lora_config["bias"],
        random_state=train_config["seed"],
    )

    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=SFTConfig(
            output_dir=train_config["output_dir"],
            max_length=model_config["max_sequence_length"],
            per_device_train_batch_size=train_config["per_device_train_batch_size"],
            per_device_eval_batch_size=train_config["per_device_eval_batch_size"],
            gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
            learning_rate=train_config["learning_rate"],
            weight_decay=train_config["weight_decay"],
            warmup_steps=train_config["warmup_steps"],
            max_steps=train_config["max_steps"],
            eval_strategy="steps",
            eval_steps=train_config["eval_steps"],
            save_strategy="steps",
            save_steps=train_config["save_steps"],
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=train_config["logging_steps"],
            optim=train_config["optimizer"],
            lr_scheduler_type=train_config["scheduler"],
            seed=train_config["seed"],
            data_seed=train_config["seed"],
            bf16=use_bf16,
            fp16=torch.cuda.is_available() and not use_bf16,
            completion_only_loss=True,
            report_to="none",
        ),
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train()
    validation = trainer.evaluate()

    adapter_dir = Path(train_config["adapter_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    run_summary = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "model": model_config["name"],
        "adapter_dir": str(adapter_dir),
        "train_examples": len(dataset["train"]),
        "validation_examples": len(dataset["validation"]),
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_parameter_percent": 100 * trainable_parameters / total_parameters,
        "train_metrics": train_result.metrics,
        "validation_metrics": validation,
        "peak_gpu_memory_gib": (
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        ),
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "config": config,
    }
    summary_path = adapter_dir / "training_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
