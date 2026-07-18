"""Local, pinned Hugging Face token-classifier training over mined span data.

Framework imports are intentionally delayed until after dataset and manifest
validation. Core installs can therefore inspect mining artifacts without importing
Torch, and model runs cannot silently download an unpinned checkpoint.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medical_kg_nlp.adapters.huggingface.runtime import OptionalModelDependencyError
from medical_kg_nlp.mining.io import write_json
from medical_kg_nlp.training.config import TokenClassifierTrainingConfig
from medical_kg_nlp.training.span_dataset import (
    SpanDatasetSummary,
    SpanTrainingRecord,
    build_bio_label_vocabulary,
    scan_span_dataset,
    validate_span_dataset_manifest,
)
from medical_kg_nlp.training.token_labels import (
    compute_bio_span_metrics,
    project_record_to_token_windows,
)
from medical_kg_nlp.utils.hashing import sha256_file, sha256_text

__all__ = [
    "inspect_token_classifier_training_inputs",
    "train_huggingface_token_classifier",
]


def inspect_token_classifier_training_inputs(
    config: TokenClassifierTrainingConfig,
) -> tuple[SpanDatasetSummary, tuple[str, ...]]:
    """Validate dataset identity, offsets, split labels, and BIO vocabulary."""

    summary = scan_span_dataset(config.dataset_path)
    validate_span_dataset_manifest(
        config.dataset_path,
        config.dataset_manifest_path,
        summary,
    )
    vocabulary = build_bio_label_vocabulary(
        summary,
        train_split=config.train_split,
        evaluation_split=(
            None
            if config.internal_validation_fraction
            else config.evaluation_split
        ),
    )
    if config.train_split not in summary.split_record_counts:
        raise ValueError(f"Training split {config.train_split!r} is absent")
    if (
        config.evaluation_split is not None
        and config.evaluation_split not in summary.split_record_counts
    ):
        raise ValueError(f"Evaluation split {config.evaluation_split!r} is absent")
    return summary, vocabulary


def train_huggingface_token_classifier(
    config: TokenClassifierTrainingConfig,
) -> Mapping[str, Any]:
    """Train, evaluate, save, and fingerprint one local token classifier."""

    summary, vocabulary = inspect_token_classifier_training_inputs(config)
    _validate_output_directory(config)
    torch, datasets, transformers = _load_training_dependencies()
    started_at = datetime.now(UTC).isoformat()

    # MODEL: both tokenizer and model resolve from a pinned local cache only.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.revision,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise ValueError("Token-classifier training requires a fast tokenizer")

    raw_dataset = datasets.load_dataset(
        "json",
        data_files=str(config.dataset_path),
        split="train",
        cache_dir=(None if config.cache_dir is None else str(config.cache_dir)),
    )
    workers = None if config.preprocessing_workers == 1 else config.preprocessing_workers
    if config.internal_validation_fraction:
        train_records = raw_dataset.filter(
            _matches_internal_train,
            input_columns=["split", "document_id"],
            fn_kwargs={
                "split_name": config.train_split,
                "fraction": config.internal_validation_fraction,
            },
            num_proc=workers,
            desc="Select internal-training records",
        )
    else:
        train_records = raw_dataset.filter(
            _matches_split,
            input_columns=["split"],
            fn_kwargs={"split_name": config.train_split},
            num_proc=workers,
            desc=f"Select {config.train_split} records",
        )
    evaluation_records = None
    if config.internal_validation_fraction:
        evaluation_records = raw_dataset.filter(
            _matches_internal_evaluation,
            input_columns=["split", "document_id"],
            fn_kwargs={
                "split_name": config.train_split,
                "fraction": config.internal_validation_fraction,
            },
            num_proc=workers,
            desc="Select internal-evaluation records",
        )
    elif config.evaluation_split is not None:
        evaluation_records = raw_dataset.filter(
            _matches_split,
            input_columns=["split"],
            fn_kwargs={"split_name": config.evaluation_split},
            num_proc=workers,
            desc=f"Select {config.evaluation_split} records",
        )

    projection_arguments = {
        "tokenizer": tokenizer,
        "label_vocabulary": vocabulary,
        "max_length": config.max_length,
        "stride": config.stride,
    }
    # SCALING: datasets writes mapped token windows to Arrow and memory-maps them.
    train_dataset = train_records.map(
        _tokenize_batch,
        batched=True,
        batch_size=64,
        remove_columns=train_records.column_names,
        fn_kwargs=projection_arguments,
        num_proc=workers,
        desc="Project training spans to BIO windows",
    )
    evaluation_dataset = None
    if evaluation_records is not None:
        evaluation_dataset = evaluation_records.map(
            _tokenize_batch,
            batched=True,
            batch_size=64,
            remove_columns=evaluation_records.column_names,
            fn_kwargs=projection_arguments,
            num_proc=workers,
            desc="Project evaluation spans to BIO windows",
        )
    if len(train_dataset) == 0:
        raise ValueError("Tokenization produced no training windows")
    if evaluation_dataset is not None and len(evaluation_dataset) == 0:
        raise ValueError("Tokenization produced no evaluation windows")

    label_to_id = {label: index for index, label in enumerate(vocabulary)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    model = transformers.AutoModelForTokenClassification.from_pretrained(
        config.model_id,
        revision=config.revision,
        local_files_only=True,
        trust_remote_code=False,
        num_labels=len(vocabulary),
        label2id=label_to_id,
        id2label=id_to_label,
        ignore_mismatched_sizes=True,
    )

    has_evaluation = evaluation_dataset is not None
    training_arguments = transformers.TrainingArguments(
        output_dir=str(config.output_dir),
        overwrite_output_dir=config.overwrite_output,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.evaluation_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        eval_strategy="epoch" if has_evaluation else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=has_evaluation,
        metric_for_best_model="span_f1" if has_evaluation else None,
        greater_is_better=True if has_evaluation else None,
        logging_strategy="steps",
        logging_steps=10,
        report_to=[],
        seed=config.seed,
        data_seed=config.seed,
        fp16=config.fp16,
        bf16=config.bf16,
        use_cpu=config.use_cpu,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=evaluation_dataset,
        data_collator=transformers.DataCollatorForTokenClassification(tokenizer),
        processing_class=tokenizer,
        compute_metrics=_metric_function(vocabulary) if has_evaluation else None,
    )
    train_result = trainer.train(
        resume_from_checkpoint=(
            None
            if config.resume_from_checkpoint is None
            else str(config.resume_from_checkpoint)
        )
    )
    evaluation_metrics = trainer.evaluate() if has_evaluation else {}

    final_model_dir = config.output_dir / "final-model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    model_fingerprint = _fingerprint_directory(final_model_dir)
    completed_at = datetime.now(UTC).isoformat()
    manifest: dict[str, Any] = {
        "schema_version": "token-classifier-training.v1",
        "started_at": started_at,
        "completed_at": completed_at,
        "model": {
            "model_id": config.model_id,
            "revision": config.revision,
            "local_files_only": True,
            "output": str(final_model_dir),
            "fingerprint": model_fingerprint,
        },
        "configuration": config.to_dict(),
        "configuration_sha256": sha256_text(
            json.dumps(config.to_dict(), ensure_ascii=False, sort_keys=True)
        ),
        "dataset": summary.to_dict(),
        "dataset_manifest_sha256": sha256_file(config.dataset_manifest_path),
        "label_vocabulary": list(vocabulary),
        "window_counts": {
            "train": len(train_dataset),
            "evaluation": 0 if evaluation_dataset is None else len(evaluation_dataset),
        },
        "metrics": {
            "train": _json_metrics(train_result.metrics),
            "evaluation": _json_metrics(evaluation_metrics),
        },
        "runtime": {
            "torch": str(getattr(torch, "__version__", "unknown")),
            "datasets": str(getattr(datasets, "__version__", "unknown")),
            "transformers": str(getattr(transformers, "__version__", "unknown")),
        },
    }
    write_json(config.output_dir / "run_manifest.json", manifest)
    return manifest


def _load_training_dependencies() -> tuple[Any, Any, Any]:
    try:
        torch = importlib.import_module("torch")
        datasets = importlib.import_module("datasets")
        transformers = importlib.import_module("transformers")
    except (ImportError, ModuleNotFoundError) as error:
        raise OptionalModelDependencyError(
            "Token-classifier training requires the local 'ml' extra: "
            "uv sync --extra ml"
        ) from error
    return torch, datasets, transformers


def _matches_split(value: object, *, split_name: str) -> bool:
    return str(value) == split_name


def _matches_internal_train(
    split: object,
    document_id: object,
    *,
    split_name: str,
    fraction: float,
) -> bool:
    return str(split) == split_name and not _is_internal_validation_document(
        str(document_id), fraction
    )


def _matches_internal_evaluation(
    split: object,
    document_id: object,
    *,
    split_name: str,
    fraction: float,
) -> bool:
    return str(split) == split_name and _is_internal_validation_document(
        str(document_id), fraction
    )


def _is_internal_validation_document(document_id: str, fraction: float) -> bool:
    # SCALING: route all chunks of a document together without materializing a split file.
    bucket = int(sha256_text(f"token-validation\x1f{document_id}")[:8], 16) / 0xFFFFFFFF
    return bucket < fraction


def _tokenize_batch(
    batch: Mapping[str, Sequence[Any]],
    *,
    tokenizer: Any,
    label_vocabulary: Sequence[str],
    max_length: int,
    stride: int,
) -> dict[str, list[list[int]]]:
    if "text" not in batch:
        raise ValueError("Tokenization batch is missing text")
    output: dict[str, list[list[int]]] = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    include_token_types: bool | None = None
    for row_index in range(len(batch["text"])):
        raw = {key: values[row_index] for key, values in batch.items()}
        record = SpanTrainingRecord.from_mapping(raw)
        windows = project_record_to_token_windows(
            record,
            tokenizer,
            label_vocabulary,
            max_length=max_length,
            stride=stride,
        )
        for window in windows:
            model_row = window.to_model_dict()
            has_token_types = "token_type_ids" in model_row
            if include_token_types is None:
                include_token_types = has_token_types
                if has_token_types:
                    output["token_type_ids"] = []
            elif include_token_types != has_token_types:
                raise ValueError("Tokenizer emitted inconsistent token_type_ids")
            for key, values in model_row.items():
                output[key].append(values)
    return output


def _metric_function(label_vocabulary: Sequence[str]) -> Any:
    def compute_metrics(evaluation_prediction: Any) -> dict[str, float]:
        predictions = evaluation_prediction.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predicted_ids = predictions.argmax(axis=-1).tolist()
        gold_ids = evaluation_prediction.label_ids.tolist()
        return compute_bio_span_metrics(predicted_ids, gold_ids, label_vocabulary)

    return compute_metrics


def _validate_output_directory(config: TokenClassifierTrainingConfig) -> None:
    output = config.output_dir
    has_existing_files = output.exists() and any(output.iterdir())
    if (
        has_existing_files
        and not config.overwrite_output
        and config.resume_from_checkpoint is None
    ):
        raise ValueError(
            f"Output directory {output} is not empty; choose a new run directory, "
            "resume a checkpoint, or pass overwrite_output"
        )
    output.mkdir(parents=True, exist_ok=True)


def _fingerprint_directory(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(value for value in path.rglob("*") if value.is_file())
    if not files:
        raise ValueError(f"Saved model directory {path} is empty")
    for file_path in files:
        relative = file_path.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(file_path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _json_metrics(values: Mapping[str, object]) -> dict[str, float | int | str]:
    output: dict[str, float | int | str] = {}
    for key, value in sorted(values.items()):
        if isinstance(value, bool):
            output[str(key)] = int(value)
        elif isinstance(value, (int, float, str)):
            output[str(key)] = value
        elif hasattr(value, "item"):
            scalar = value.item()
            if isinstance(scalar, (int, float, str)):
                output[str(key)] = scalar
    return output
