"""Local, pinned Hugging Face token-classifier training over mined span data.

Framework imports are intentionally delayed until after dataset and manifest
validation. Core installs can therefore inspect mining artifacts without importing
Torch, and model runs cannot silently download an unpinned checkpoint.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medical_kg_nlp.adapters.huggingface.runtime import OptionalModelDependencyError
from medical_kg_nlp.adapters.huggingface.config import HuggingFaceModelConfig
from medical_kg_nlp.adapters.huggingface.token_classifier import (
    HuggingFaceTokenClassifierAdapter,
)
from medical_kg_nlp.mining.io import write_json
from medical_kg_nlp.training.config import TokenClassifierTrainingConfig
from medical_kg_nlp.training.span_dataset import (
    SpanDatasetSummary,
    SpanTrainingRecord,
    build_bio_label_vocabulary,
    iter_span_training_records,
    scan_span_dataset,
    validate_span_dataset_manifest,
)
from medical_kg_nlp.training.token_labels import (
    TokenAlignmentPolicy,
    TokenBoundaryAlignmentError,
    compute_bio_span_metrics,
    find_unaligned_annotations,
    project_record_to_token_windows,
)
from medical_kg_nlp.utils.hashing import sha256_file, sha256_text

__all__ = [
    "fingerprint_model_directory",
    "inspect_token_classifier_training_inputs",
    "train_huggingface_token_classifier",
    "verify_token_classifier_artifact",
    "verify_saved_token_classifier",
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
        evaluation_split=(None if config.internal_validation_fraction else config.evaluation_split),
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
    *,
    manifest_root: Path | None = None,
) -> Mapping[str, Any]:
    """Train, evaluate, save, and fingerprint one local token classifier.

    A checked-in run spec passes ``manifest_root`` so persisted paths remain
    repository-relative even though runtime IO uses fully resolved paths.
    """

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
    alignment_report = _inspect_token_alignment(
        config,
        tokenizer,
        entity_count=summary.entity_count,
    )
    if alignment_report["ignored_annotation_count"] and config.unaligned_span_policy == "error":
        raise TokenBoundaryAlignmentError(
            "Tokenizer cannot preserve every exact annotation boundary; "
            "inspect the alignment report or explicitly use unaligned_span_policy='mask'"
        )

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
        "unaligned_span_policy": config.unaligned_span_policy,
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
        full_determinism=config.full_determinism,
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
            None if config.resume_from_checkpoint is None else str(config.resume_from_checkpoint)
        )
    )
    evaluation_metrics = trainer.evaluate() if has_evaluation else {}

    final_model_dir = config.output_dir / "final-model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    model_fingerprint = fingerprint_model_directory(final_model_dir)
    completed_at = datetime.now(UTC).isoformat()
    manifest_configuration = config.to_dict(path_root=manifest_root)
    manifest: dict[str, Any] = {
        "schema_version": "token-classifier-training.v2",
        "started_at": started_at,
        "completed_at": completed_at,
        "path_contract": {
            "base": "run_root" if manifest_root is not None else "literal",
        },
        "model": {
            "model_id": config.model_id,
            "revision": config.revision,
            "local_files_only": True,
            "output": _manifest_path(final_model_dir, root=manifest_root),
            "fingerprint": model_fingerprint,
        },
        "configuration": manifest_configuration,
        "configuration_sha256": sha256_text(
            json.dumps(manifest_configuration, ensure_ascii=False, sort_keys=True)
        ),
        "dataset": summary.to_dict(),
        "dataset_manifest_sha256": sha256_file(config.dataset_manifest_path),
        "label_vocabulary": list(vocabulary),
        "token_alignment": alignment_report,
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


def verify_saved_token_classifier(
    model_dir: str | Path,
    source_text: str,
    *,
    max_length: int = 512,
    stride: int = 64,
    batch_size: int = 1,
) -> Mapping[str, Any]:
    """Reload a saved model and prove its emitted spans use raw-text coordinates.

    This is intentionally an inference check rather than another metric pass. A CPU smoke job
    calls it after ``Trainer.save_model`` so tokenizer serialization, weight serialization,
    overflow inference, and offset projection all execute in a fresh adapter runtime.
    """

    if not source_text:
        raise ValueError("CPU smoke verification text must be non-empty")
    path = Path(model_dir).resolve()
    if not path.is_dir():
        raise ValueError(f"Saved token-classifier directory does not exist: {path}")
    adapter = HuggingFaceTokenClassifierAdapter(
        HuggingFaceModelConfig(
            model_id=str(path),
            # MODEL: local directories are content-fingerprinted by the training manifest;
            # revision remains explicit to satisfy the adapter provenance contract.
            revision="local-saved-artifact",
            device="cpu",
            batch_size=batch_size,
            max_length=max_length,
        ),
        stride=stride,
    )
    entities = adapter.extract(source_text)
    for entity in entities:
        # INVARIANT: a saved/reloaded model is accepted only if every projected span round-trips.
        entity.validate_offsets(source_text)
    return {
        "status": "passed",
        "source_text_sha256": sha256_text(source_text),
        "source_character_count": len(source_text),
        "projected_entity_count": len(entities),
        "offset_mismatch_count": 0,
        "model_directory": str(path),
    }


def _load_training_dependencies() -> tuple[Any, Any, Any]:
    try:
        torch = importlib.import_module("torch")
        datasets = importlib.import_module("datasets")
        transformers = importlib.import_module("transformers")
    except (ImportError, ModuleNotFoundError) as error:
        raise OptionalModelDependencyError(
            "Token-classifier training requires the local 'ml' extra: uv sync --extra ml"
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
    return str(split) == split_name and _is_internal_validation_document(str(document_id), fraction)


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
    unaligned_span_policy: TokenAlignmentPolicy,
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
            unaligned_span_policy=unaligned_span_policy,
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


def _inspect_token_alignment(
    config: TokenClassifierTrainingConfig,
    tokenizer: Any,
    *,
    entity_count: int,
) -> dict[str, Any]:
    """Audit tokenizer-only label loss before expensive framework training starts."""

    by_split: Counter[str] = Counter()
    by_label: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for record in iter_span_training_records(config.dataset_path):
        issues = find_unaligned_annotations(
            record,
            tokenizer,
            max_length=config.max_length,
            stride=config.stride,
        )
        for entity in issues:
            by_split[record.split] += 1
            by_label[entity.label] += 1
            if len(examples) < 20:
                examples.append(
                    {
                        "record_id": record.record_id,
                        "document_id": record.document_id,
                        "annotation_id": entity.annotation_id,
                        "label": entity.label,
                        "span": [entity.start, entity.end],
                        "text": entity.text,
                    }
                )
    count = sum(by_split.values())
    return {
        "policy": config.unaligned_span_policy,
        "ignored_annotation_count": count,
        "ignored_fraction": count / max(1, entity_count),
        "by_split": dict(sorted(by_split.items())),
        "by_label": dict(sorted(by_label.items())),
        "examples": examples,
    }


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
    if has_existing_files and not config.overwrite_output and config.resume_from_checkpoint is None:
        raise ValueError(
            f"Output directory {output} is not empty; choose a new run directory, "
            "resume a checkpoint, or pass overwrite_output"
        )
    output.mkdir(parents=True, exist_ok=True)


def fingerprint_model_directory(path: str | Path) -> str:
    """Hash a saved model by relative file path and file content."""

    root = Path(path)
    digest = hashlib.sha256()
    files = sorted(value for value in root.rglob("*") if value.is_file())
    if not files:
        raise ValueError(f"Saved model directory {root} is missing or empty")
    for file_path in files:
        relative = file_path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(file_path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_token_classifier_artifact(
    model_dir: str | Path,
    manifest_path: str | Path,
) -> Mapping[str, str]:
    """Verify transferred model bytes against their immutable training manifest."""

    resolved_model_dir = Path(model_dir)
    resolved_manifest_path = Path(manifest_path)
    if not resolved_manifest_path.is_file():
        raise ValueError(
            f"Token-classifier training manifest does not exist: {resolved_manifest_path}"
        )
    payload = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Token-classifier training manifest must be a JSON object")
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Token-classifier training manifest is missing model metadata")
    expected_fingerprint = model.get("fingerprint")
    if not isinstance(expected_fingerprint, str) or not expected_fingerprint:
        raise ValueError("Token-classifier training manifest is missing model fingerprint")
    actual_fingerprint = fingerprint_model_directory(resolved_model_dir)
    if actual_fingerprint != expected_fingerprint:
        raise ValueError(
            "Saved model fingerprint mismatch: "
            f"expected {expected_fingerprint}, got {actual_fingerprint}"
        )
    return {
        "status": "verified",
        "manifest": str(resolved_manifest_path),
        "model": str(resolved_model_dir),
        "fingerprint": actual_fingerprint,
        "manifest_sha256": sha256_file(resolved_manifest_path),
    }


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


def _manifest_path(path: Path, *, root: Path | None) -> str:
    if root is None:
        return str(path)
    try:
        # INVARIANT: model provenance may move between hosts as one release tree.
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Model output escapes manifest root: {path}") from error
