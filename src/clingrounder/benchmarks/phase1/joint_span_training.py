"""Train a pinned transformer verifier for Phase 1 span/type candidate lattices.

The training input is deliberately proposal-derived rather than a gold span list. The model
therefore learns to distinguish exact candidates from nearby short/long alternatives and from
spurious proposals without synthesizing a new offset at inference time.
"""

from __future__ import annotations

import importlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clingrounder.benchmarks.phase1.joint_span import Phase1JointSpanLabel
from clingrounder.mining.io import write_json
from clingrounder.training.huggingface_token_classifier import fingerprint_model_directory
from clingrounder.utils.hashing import sha256_file, sha256_text

__all__ = [
    "Phase1JointSpanTrainingConfig",
    "Phase1JointSpanTrainingSummary",
    "inspect_phase1_joint_span_training_inputs",
    "phase1_joint_span_training_family_fingerprint",
    "train_phase1_joint_span_verifier",
    "verify_phase1_joint_span_verifier_artifact",
]

_DATASET_SCHEMA = "phase1-joint-span-dataset.v1"
_TRAINING_ARTIFACT_SCHEMA = "phase1-joint-span-verifier-training.v2"
_LABELS = tuple(label.value for label in Phase1JointSpanLabel)


@dataclass(frozen=True, slots=True)
class Phase1JointSpanTrainingConfig:
    """Immutable inputs and bounded optimization settings for one verifier fit."""

    dataset_path: Path
    dataset_manifest_path: Path
    output_dir: Path
    model_id: str
    revision: str
    initialization_model_path: Path | None = None
    initialization_model_fingerprint: str | None = None
    training_family_dataset_sha256: str | None = None
    max_length: int = 384
    train_batch_size: int = 8
    evaluation_batch_size: int = 16
    epochs: float = 4.0
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.08
    gradient_accumulation_steps: int = 1
    seed: int = 42
    fp16: bool = False
    bf16: bool = False
    use_cpu: bool = False
    cache_dir: Path | None = None
    overwrite_output: bool = False

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not self.revision.strip():
            raise ValueError("Joint span training requires a pinned model ID and revision")
        if (self.initialization_model_path is None) != (
            self.initialization_model_fingerprint is None
        ):
            raise ValueError(
                "initialization_model_path and initialization_model_fingerprint must be paired"
            )
        if self.initialization_model_fingerprint is not None and not _is_sha256(
            self.initialization_model_fingerprint
        ):
            raise ValueError("initialization_model_fingerprint must be a lowercase SHA-256")
        if self.training_family_dataset_sha256 is not None and not _is_sha256(
            self.training_family_dataset_sha256
        ):
            raise ValueError("training_family_dataset_sha256 must be a lowercase SHA-256")
        if self.max_length < 32:
            raise ValueError("Joint span verifier max_length must be at least 32")
        if self.train_batch_size < 1 or self.evaluation_batch_size < 1:
            raise ValueError("Joint span verifier batch sizes must be positive")
        if self.epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("Joint span verifier epochs and learning_rate must be positive")
        if self.weight_decay < 0 or not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("Joint span verifier optimizer settings are invalid")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.fp16 and self.bf16:
            raise ValueError("fp16 and bf16 cannot both be enabled")

    def to_dict(self) -> dict[str, Any]:
        """Serialize every setting which affects trained model bytes."""

        return {
            "dataset_path": str(self.dataset_path),
            "dataset_manifest_path": str(self.dataset_manifest_path),
            "output_dir": str(self.output_dir),
            "model_id": self.model_id,
            "revision": self.revision,
            "initialization_model_path": (
                None
                if self.initialization_model_path is None
                else str(self.initialization_model_path)
            ),
            "initialization_model_fingerprint": self.initialization_model_fingerprint,
            "training_family_dataset_sha256": self.training_family_dataset_sha256,
            "max_length": self.max_length,
            "train_batch_size": self.train_batch_size,
            "evaluation_batch_size": self.evaluation_batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "seed": self.seed,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "use_cpu": self.use_cpu,
            "cache_dir": None if self.cache_dir is None else str(self.cache_dir),
            "overwrite_output": self.overwrite_output,
        }


@dataclass(frozen=True, slots=True)
class Phase1JointSpanTrainingSummary:
    """Validated immutable identity of a proposal-derived verifier dataset."""

    dataset_sha256: str
    example_count: int
    document_count: int
    label_counts: Mapping[str, int]
    source_dataset_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return the stable manifest fragment saved beside a trained checkpoint."""

        return {
            "dataset_sha256": self.dataset_sha256,
            "example_count": self.example_count,
            "document_count": self.document_count,
            "label_counts": dict(sorted(self.label_counts.items())),
            "source_dataset_counts": dict(sorted(self.source_dataset_counts.items())),
        }


def phase1_joint_span_training_family_fingerprint(
    config: Phase1JointSpanTrainingConfig,
    summary: Phase1JointSpanTrainingSummary,
) -> str:
    """Pin the shared OOF/final-fit training contract, excluding checkpoint byte identity.

    MODEL: an OOF fold must be trained on a subset, so it cannot share the final model SHA-256.
    This fingerprint binds both artifacts to the same full supervision corpus, initializer, label
    space, and optimization recipe while keeping runtime paths and output locations irrelevant.
    """

    payload = {
        "schema_version": "phase1-joint-span-training-family.v1",
        "supervision_dataset_sha256": (
            config.training_family_dataset_sha256 or summary.dataset_sha256
        ),
        "model_id": config.model_id,
        "revision": config.revision,
        "initialization_model_fingerprint": config.initialization_model_fingerprint,
        "max_length": config.max_length,
        "train_batch_size": config.train_batch_size,
        "evaluation_batch_size": config.evaluation_batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "seed": config.seed,
        "labels": list(_LABELS),
        "loss_weighting": "inverse_frequency_capped_6",
    }
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def inspect_phase1_joint_span_training_inputs(
    config: Phase1JointSpanTrainingConfig,
) -> Phase1JointSpanTrainingSummary:
    """Validate all inputs before importing ML dependencies or allocating a GPU.

    INVARIANT: every input row must still contain the immutable raw candidate substring. The
    verifier never consumes a generated or normalized replacement span.
    """

    if not config.dataset_path.is_file() or not config.dataset_manifest_path.is_file():
        raise ValueError("Joint span dataset and manifest must exist")
    payload = _load_mapping(config.dataset_manifest_path, "joint span dataset manifest")
    if payload.get("schema_version") != _DATASET_SCHEMA:
        raise ValueError("Unsupported joint span dataset schema")
    expected_sha256 = payload.get("examples_sha256")
    if not isinstance(expected_sha256, str) or not _is_sha256(expected_sha256):
        raise ValueError("Joint span manifest examples_sha256 is invalid")
    actual_sha256 = sha256_file(config.dataset_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("Joint span example file does not match its manifest")

    labels: Counter[str] = Counter()
    source_datasets: Counter[str] = Counter()
    document_ids: set[str] = set()
    variant_ids: set[str] = set()
    example_count = 0
    for row_number, row in _iter_examples(config.dataset_path):
        label = _required_string(row, "label")
        if label not in _LABELS:
            raise ValueError(f"{config.dataset_path}:{row_number}: unsupported joint label")
        variant_id = _required_string(row, "variant_id")
        if variant_id in variant_ids:
            raise ValueError(f"{config.dataset_path}:{row_number}: duplicate variant_id")
        variant_ids.add(variant_id)
        document_ids.add(_required_string(row, "document_id"))
        source_datasets[_required_string(row, "source_dataset")] += 1
        cross_encoder_text = _required_string(row, "cross_encoder_text")
        if not cross_encoder_text.strip():
            raise ValueError(f"{config.dataset_path}:{row_number}: empty cross_encoder_text")
        text = _required_string(row, "text")
        position = row.get("position")
        if (
            not isinstance(position, list)
            or len(position) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in position)
            or position[0] < 0
            or position[1] <= position[0]
            or not text
        ):
            raise ValueError(f"{config.dataset_path}:{row_number}: invalid raw candidate identity")
        labels[label] += 1
        example_count += 1
    if not example_count:
        raise ValueError("Joint span training dataset is empty")
    if set(labels) != set(_LABELS):
        missing = sorted(set(_LABELS) - set(labels))
        raise ValueError(f"Joint span training data is missing labels: {missing}")
    return Phase1JointSpanTrainingSummary(
        dataset_sha256=actual_sha256,
        example_count=example_count,
        document_count=len(document_ids),
        label_counts=dict(labels),
        source_dataset_counts=dict(source_datasets),
    )


def train_phase1_joint_span_verifier(
    config: Phase1JointSpanTrainingConfig,
) -> Mapping[str, Any]:
    """Fine-tune a local cross encoder and persist a fingerprinted inference artifact.

    The full supervision corpus is intentionally used for the final fit. Diagnostic validation
    is not a model-promotion gate; official BTC submission metrics decide promotion.
    """

    summary = inspect_phase1_joint_span_training_inputs(config)
    training_family_fingerprint = phase1_joint_span_training_family_fingerprint(config, summary)
    _validate_output_directory(config)
    initialization = _initialization_model(config)
    torch, datasets, transformers = _load_training_dependencies()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        initialization,
        revision=None if config.initialization_model_path is not None else config.revision,
        local_files_only=True,
        use_fast=True,
    )
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise ValueError("Joint span verifier requires a fast tokenizer")
    raw_dataset = datasets.load_dataset(
        "json",
        data_files=str(config.dataset_path),
        split="train",
        cache_dir=None if config.cache_dir is None else str(config.cache_dir),
    )
    label_to_id = {label: index for index, label in enumerate(_LABELS)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    encoded_dataset = raw_dataset.map(
        _tokenize_examples,
        batched=True,
        remove_columns=raw_dataset.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label_to_id": label_to_id,
            "max_length": config.max_length,
        },
        desc="Tokenize joint span lattice examples",
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        initialization,
        revision=None if config.initialization_model_path is not None else config.revision,
        local_files_only=True,
        trust_remote_code=False,
        num_labels=len(_LABELS),
        label2id=label_to_id,
        id2label=id_to_label,
        ignore_mismatched_sizes=True,
    )
    # MODEL: inverse-frequency weights keep rare exact drug/result evidence visible without
    # duplicating raw text or allowing a synthetic label to enter final supervision.
    loss_weights = _class_loss_weights(summary.label_counts, torch)
    trainer = _weighted_trainer(
        transformers,
        torch,
        model=model,
        arguments=transformers.TrainingArguments(
            output_dir=str(config.output_dir),
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.train_batch_size,
            per_device_eval_batch_size=config.evaluation_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            eval_strategy="no",
            save_strategy="epoch",
            save_total_limit=2,
            save_only_model=True,
            logging_strategy="steps",
            logging_steps=20,
            report_to=[],
            seed=config.seed,
            data_seed=config.seed,
            fp16=config.fp16,
            bf16=config.bf16,
            use_cpu=config.use_cpu,
        ),
        train_dataset=encoded_dataset,
        processing_class=tokenizer,
        data_collator=transformers.DataCollatorWithPadding(tokenizer),
        loss_weights=loss_weights,
    )
    result = trainer.train()
    final_model = config.output_dir / "final-model"
    trainer.save_model(str(final_model))
    tokenizer.save_pretrained(str(final_model))
    fingerprint = fingerprint_model_directory(final_model)
    manifest = {
        "schema_version": _TRAINING_ARTIFACT_SCHEMA,
        "purpose": "final_fit_all_authorized_supervision",
        "promotion": "official_submission_metrics_only",
        "model": {
            "model_id": config.model_id,
            "revision": config.revision,
            "initialization": (
                {"kind": "huggingface_cache"}
                if config.initialization_model_path is None
                else {
                    "kind": "local_artifact",
                    "path": str(config.initialization_model_path),
                    "fingerprint": config.initialization_model_fingerprint,
                }
            ),
            "output": str(final_model),
            "fingerprint": fingerprint,
            "labels": list(_LABELS),
        },
        "dataset": summary.to_dict(),
        "training_family_fingerprint": training_family_fingerprint,
        "dataset_manifest_sha256": sha256_file(config.dataset_manifest_path),
        "configuration": config.to_dict(),
        "configuration_sha256": sha256_text(
            json.dumps(config.to_dict(), ensure_ascii=False, sort_keys=True)
        ),
        "optimization": {
            "class_loss_weights": {
                label: float(loss_weights[index].detach().cpu().item())
                for index, label in enumerate(_LABELS)
            },
            "train_metrics": _json_metrics(result.metrics),
        },
        "runtime": {
            "torch": str(getattr(torch, "__version__", "unknown")),
            "datasets": str(getattr(datasets, "__version__", "unknown")),
            "transformers": str(getattr(transformers, "__version__", "unknown")),
        },
    }
    write_json(config.output_dir / "run_manifest.json", manifest)
    return manifest


def verify_phase1_joint_span_verifier_artifact(
    config: Phase1JointSpanTrainingConfig,
) -> Mapping[str, Any]:
    """Verify final model bytes and immutable training inputs without importing Torch."""

    summary = inspect_phase1_joint_span_training_inputs(config)
    manifest_path = config.output_dir / "run_manifest.json"
    manifest = _load_mapping(manifest_path, "joint span training manifest")
    if manifest.get("schema_version") != _TRAINING_ARTIFACT_SCHEMA:
        raise ValueError("Unsupported joint span verifier artifact schema")
    if manifest.get("dataset") != summary.to_dict():
        raise ValueError("Joint span verifier dataset identity does not match")
    if manifest.get("training_family_fingerprint") != phase1_joint_span_training_family_fingerprint(
        config, summary
    ):
        raise ValueError("Joint span verifier training family does not match")
    if manifest.get("dataset_manifest_sha256") != sha256_file(config.dataset_manifest_path):
        raise ValueError("Joint span verifier dataset manifest does not match")
    model = _mapping(manifest.get("model"), "joint span verifier model")
    if model.get("model_id") != config.model_id or model.get("revision") != config.revision:
        raise ValueError("Joint span verifier model identity does not match")
    if model.get("labels") != list(_LABELS):
        raise ValueError("Joint span verifier labels do not match the runtime contract")
    model_dir = config.output_dir / "final-model"
    actual_fingerprint = fingerprint_model_directory(model_dir)
    if model.get("fingerprint") != actual_fingerprint:
        raise ValueError("Joint span verifier model fingerprint does not match")
    return {
        "status": "verified",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "model": str(model_dir),
        "fingerprint": actual_fingerprint,
        "dataset": summary.to_dict(),
    }


def _iter_examples(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{row_number}: invalid JSON") from error
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{row_number}: expected object")
            yield row_number, row


def _tokenize_examples(
    rows: Mapping[str, Sequence[object]],
    *,
    tokenizer: Any,
    label_to_id: Mapping[str, int],
    max_length: int,
) -> dict[str, Any]:
    texts = [str(value) for value in rows["cross_encoder_text"]]
    labels = [label_to_id[str(value)] for value in rows["label"]]
    encoded = tokenizer(texts, truncation=True, max_length=max_length)
    encoded["labels"] = labels
    return dict(encoded)


def _weighted_trainer(
    transformers: Any,
    torch: Any,
    *,
    model: Any,
    arguments: Any,
    train_dataset: Any,
    processing_class: Any,
    data_collator: Any,
    loss_weights: Any,
) -> Any:
    class WeightedSequenceTrainer(transformers.Trainer):  # type: ignore[misc]
        def compute_loss(self, model: Any, inputs: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            weights = loss_weights.to(outputs.logits.device)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weights)
            if kwargs.get("return_outputs", False):
                return loss, outputs
            return loss

    return WeightedSequenceTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        processing_class=processing_class,
        data_collator=data_collator,
    )


def _class_loss_weights(label_counts: Mapping[str, int], torch: Any) -> Any:
    total = sum(label_counts.values())
    class_count = len(_LABELS)
    # A cap prevents the least frequent label from dominating an entire batch.
    values = [min(6.0, total / (class_count * label_counts[label])) for label in _LABELS]
    return torch.tensor(values, dtype=torch.float32)


def _initialization_model(config: Phase1JointSpanTrainingConfig) -> str:
    if config.initialization_model_path is None:
        return config.model_id
    if not config.initialization_model_path.is_dir():
        raise ValueError("Joint span initializer directory does not exist")
    if fingerprint_model_directory(config.initialization_model_path) != config.initialization_model_fingerprint:
        raise ValueError("Joint span initializer fingerprint does not match")
    return str(config.initialization_model_path)


def _validate_output_directory(config: Phase1JointSpanTrainingConfig) -> None:
    if config.output_dir.exists() and any(config.output_dir.iterdir()) and not config.overwrite_output:
        raise ValueError("Joint span output directory exists; pass overwrite_output to replace it")
    config.output_dir.mkdir(parents=True, exist_ok=True)


def _load_training_dependencies() -> tuple[Any, Any, Any]:
    try:
        return (
            importlib.import_module("torch"),
            importlib.import_module("datasets"),
            importlib.import_module("transformers"),
        )
    except ImportError as error:
        raise RuntimeError(
            "Joint span training requires the optional 'ml' dependencies; install with uv sync --extra ml"
        ) from error


def _load_mapping(path: Path, name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"{name} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not valid JSON") from error
    return _mapping(value, name)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _required_string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _json_metrics(values: Mapping[str, object]) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in values.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
