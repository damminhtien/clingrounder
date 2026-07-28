"""Optional Linux/CUDA QLoRA runtime for chat-style causal supervision."""

from __future__ import annotations

import importlib
import importlib.util
import platform
from pathlib import Path
from typing import Any

from medical_kg_nlp.training.causal_instruction import (
    CausalInstructionRecord,
    InstructionTooLongError,
    load_causal_instruction_records,
    tokenize_causal_instruction,
)
from medical_kg_nlp.training.causal_run_spec import CausalQLoRAConfig
from medical_kg_nlp.mining.io import write_json
from medical_kg_nlp.utils.hashing import sha256_file
from medical_kg_nlp.utils.run_output import collect_git_metadata

__all__ = ["inspect_causal_qlora_inputs", "train_causal_qlora"]


def inspect_causal_qlora_inputs(config: CausalQLoRAConfig) -> dict[str, Any]:
    """Validate source fingerprints and report the deterministic selection."""

    train, train_report = load_causal_instruction_records(
        config.train_sources,
        sample_seed=f"{config.sample_seed}:train",
    )
    evaluation: list[CausalInstructionRecord] = []
    evaluation_report: dict[str, Any] | None = None
    if config.evaluation_sources:
        evaluation, report = load_causal_instruction_records(
            config.evaluation_sources,
            sample_seed=f"{config.sample_seed}:evaluation",
        )
        evaluation_report = report.to_dict()
    return {
        "train": train_report.to_dict(),
        "evaluation": evaluation_report,
        "train_record_count": len(train),
        "evaluation_record_count": len(evaluation),
        "initial_adapter_present": (
            config.initial_adapter_path is None
            or config.initial_adapter_path.is_dir()
        ),
    }


def train_causal_qlora(
    config: CausalQLoRAConfig,
    *,
    resume_from_checkpoint: Path | None = None,
    max_steps_override: int | None = None,
    output_dir_override: Path | None = None,
) -> dict[str, Any]:
    """Train one QLoRA stage and save a resumable adapter plus manifest."""

    dependencies = _training_dependencies()
    torch = dependencies["torch"]
    transformers = dependencies["transformers"]
    peft = dependencies["peft"]
    if platform.system().lower() != "linux" or not torch.cuda.is_available():
        raise RuntimeError("Causal QLoRA requires a Linux CUDA runtime")

    output_dir = (
        config.output_dir
        if output_dir_override is None
        else output_dir_override.resolve()
    )
    max_steps = config.max_steps if max_steps_override is None else max_steps_override
    if max_steps == 0:
        raise ValueError("max_steps override cannot be zero")
    output_dir.mkdir(parents=True, exist_ok=True)
    final_adapter = output_dir / "final-adapter"

    train_records, train_report = load_causal_instruction_records(
        config.train_sources,
        sample_seed=f"{config.sample_seed}:train",
    )
    evaluation_records: list[CausalInstructionRecord] = []
    evaluation_report: dict[str, Any] | None = None
    if config.evaluation_sources:
        evaluation_records, report = load_causal_instruction_records(
            config.evaluation_sources,
            sample_seed=f"{config.sample_seed}:evaluation",
        )
        evaluation_report = report.to_dict()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.revision,
        local_files_only=config.local_files_only,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_dataset, train_token_report = _tokenize_records(
        tokenizer,
        train_records,
        max_length=config.max_length,
    )
    evaluation_dataset, evaluation_token_report = _tokenize_records(
        tokenizer,
        evaluation_records,
        max_length=config.max_length,
    )

    quantization = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.revision,
        local_files_only=config.local_files_only,
        trust_remote_code=False,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model = peft.prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.gradient_checkpointing,
    )
    if config.initial_adapter_path is None:
        model = peft.get_peft_model(
            model,
            peft.LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules="all-linear",
            ),
        )
    else:
        if not config.initial_adapter_path.is_dir():
            raise ValueError(
                f"Initial QLoRA adapter is absent: {config.initial_adapter_path}"
            )
        model = peft.PeftModel.from_pretrained(
            model,
            config.initial_adapter_path,
            is_trainable=True,
        )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    has_evaluation = bool(evaluation_dataset)
    training_arguments = transformers.TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.evaluation_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs,
        max_steps=max_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        eval_strategy="steps" if has_evaluation else "no",
        eval_steps=config.evaluation_steps if has_evaluation else None,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=config.gradient_checkpointing,
        report_to=[],
        remove_unused_columns=False,
        seed=config.seed,
        data_seed=config.seed,
        full_determinism=True,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_arguments,
        train_dataset=_ListDataset(train_dataset),
        eval_dataset=(
            _ListDataset(evaluation_dataset) if has_evaluation else None
        ),
        data_collator=_CausalDataCollator(tokenizer.pad_token_id, torch),
    )
    result = trainer.train(
        resume_from_checkpoint=(
            None
            if resume_from_checkpoint is None
            else str(resume_from_checkpoint)
        )
    )
    trainer.save_state()
    model.save_pretrained(final_adapter, safe_serialization=True)
    tokenizer.save_pretrained(final_adapter)
    metrics = dict(result.metrics)
    if has_evaluation:
        metrics.update(
            {
                f"final_{key}": value
                for key, value in trainer.evaluate().items()
            }
        )

    manifest = {
        "schema_version": "causal-qlora-artifact.v1",
        "model": {
            "model_id": config.model_id,
            "revision": config.revision,
            "parameter_count": config.parameter_count,
            "initial_adapter_path": (
                None
                if config.initial_adapter_path is None
                else str(config.initial_adapter_path)
            ),
        },
        "training": {
            **_config_dict(config),
            "train_sources": [
                _source_dict(source) for source in config.train_sources
            ],
            "evaluation_sources": [
                _source_dict(source)
                for source in config.evaluation_sources
            ],
            "output_dir": str(output_dir),
            "initial_adapter_path": (
                None
                if config.initial_adapter_path is None
                else str(config.initial_adapter_path)
            ),
            "max_steps_effective": max_steps,
        },
        "datasets": {
            "train": train_report.to_dict(),
            "evaluation": evaluation_report,
            "train_tokens": train_token_report,
            "evaluation_tokens": evaluation_token_report,
        },
        "metrics": metrics,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "gpu_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "source_control": collect_git_metadata(),
        "artifacts": {
            "adapter_dir": str(final_adapter),
            "adapter_config_sha256": sha256_file(
                final_adapter / "adapter_config.json"
            ),
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def _config_dict(config: CausalQLoRAConfig) -> dict[str, Any]:
    """Serialize scalar training controls without leaking non-JSON Path objects."""

    return {
        "sample_seed": config.sample_seed,
        "max_length": config.max_length,
        "train_batch_size": config.train_batch_size,
        "evaluation_batch_size": config.evaluation_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "epochs": config.epochs,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "max_grad_norm": config.max_grad_norm,
        "logging_steps": config.logging_steps,
        "evaluation_steps": config.evaluation_steps,
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "seed": config.seed,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "gradient_checkpointing": config.gradient_checkpointing,
        "local_files_only": config.local_files_only,
    }


def _tokenize_records(
    tokenizer: Any,
    records: list[CausalInstructionRecord],
    *,
    max_length: int,
) -> tuple[list[dict[str, list[int]]], dict[str, Any]]:
    tokenized: list[dict[str, list[int]]] = []
    lengths: list[int] = []
    failures: list[str] = []
    for record in records:
        try:
            row = tokenize_causal_instruction(
                tokenizer,
                record,
                max_length=max_length,
            )
        except InstructionTooLongError as error:
            failures.append(str(error))
            continue
        tokenized.append(row)
        lengths.append(len(row["input_ids"]))
    if failures:
        raise InstructionTooLongError(
            f"{len(failures)} instruction records exceed max_length; "
            f"examples={failures[:5]}"
        )
    if records and not tokenized:
        raise ValueError("Instruction tokenization produced no records")
    return tokenized, {
        "record_count": len(tokenized),
        "minimum": min(lengths, default=0),
        "maximum": max(lengths, default=0),
        "mean": (
            sum(lengths) / len(lengths) if lengths else 0.0
        ),
    }


class _ListDataset:
    def __init__(self, rows: list[dict[str, list[int]]]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self._rows[index]


class _CausalDataCollator:
    """Right-pad causal records while preserving the -100 loss mask."""

    def __init__(self, pad_token_id: int, torch: Any) -> None:
        self._pad_token_id = pad_token_id
        self._torch = torch

    def __call__(
        self,
        features: list[dict[str, list[int]]],
    ) -> dict[str, Any]:
        maximum = max(len(feature["input_ids"]) for feature in features)
        batch: dict[str, list[list[int]]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for feature in features:
            padding = maximum - len(feature["input_ids"])
            batch["input_ids"].append(
                feature["input_ids"] + [self._pad_token_id] * padding
            )
            batch["attention_mask"].append(
                feature["attention_mask"] + [0] * padding
            )
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {
            key: self._torch.tensor(value, dtype=self._torch.long)
            for key, value in batch.items()
        }


def _source_dict(source: Any) -> dict[str, Any]:
    return {
        "path": str(source.path),
        "sha256": source.sha256,
        "split": source.split,
        "maximum_records": source.maximum_records,
        "repeat": source.repeat,
        "document_id_prefix": source.document_id_prefix,
    }


def _training_dependencies() -> dict[str, Any]:
    missing = [
        name
        for name in ("torch", "transformers", "peft", "bitsandbytes")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError(
            "Causal QLoRA requires the ML extra with PEFT/bitsandbytes; "
            f"missing={missing}"
        )
    return {
        name: importlib.import_module(name)
        for name in ("torch", "transformers", "peft", "bitsandbytes")
    }
