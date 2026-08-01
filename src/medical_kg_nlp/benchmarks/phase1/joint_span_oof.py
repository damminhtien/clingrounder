"""Document-grouped transformer OOF scoring for the Phase 1 joint span verifier.

The final joint verifier is fitted on all authorized supervision. Its probabilities cannot train
the resolver calibration because it has already seen every document. This module cross-fits
independent local cross encoders and persists only candidate identities, labels, and probabilities
for calibration; no raw note text is written to the OOF observation artifact.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from medical_kg_nlp.adapters.huggingface.config import HuggingFaceModelConfig
from medical_kg_nlp.benchmarks.phase1.joint_span import (
    Phase1JointSpanCandidate,
    Phase1JointSpanLabel,
)
from medical_kg_nlp.benchmarks.phase1.joint_span_calibration import (
    Phase1JointSpanCalibrationObservation,
)
from medical_kg_nlp.benchmarks.phase1.joint_span_training import (
    Phase1JointSpanTrainingConfig,
    inspect_phase1_joint_span_training_inputs,
    phase1_joint_span_training_family_fingerprint,
    train_phase1_joint_span_verifier,
    verify_phase1_joint_span_verifier_artifact,
)
from medical_kg_nlp.benchmarks.phase1.joint_span_verifier import (
    HuggingFacePhase1JointSpanVerifier,
)
from medical_kg_nlp.benchmarks.phase1.boundary_variants import Phase1BoundaryVariant
from medical_kg_nlp.benchmarks.phase1.proposal_features import Phase1GenreBucket
from medical_kg_nlp.mining.io import write_json, write_jsonl
from medical_kg_nlp.utils.hashing import sha256_file

__all__ = [
    "Phase1JointSpanOofConfig",
    "assign_phase1_joint_span_oof_folds",
    "run_phase1_joint_span_transformer_oof",
]

_OOF_SCHEMA = "phase1-joint-span-transformer-oof.v1"
_FOLD_ASSIGNMENT_SCHEMA = "phase1-joint-span-oof-fold-assignment.v1"
_FOLD_DATASET_SCHEMA = "phase1-joint-span-dataset.v1"
_LABELS = tuple(label.value for label in Phase1JointSpanLabel)


@dataclass(frozen=True, slots=True)
class Phase1JointSpanOofConfig:
    """Pinned base training recipe and output controls for document-grouped cross fitting."""

    training: Phase1JointSpanTrainingConfig
    output_dir: Path
    fold_count: int = 5
    inference_device: str = "cuda"
    resume: bool = False

    def __post_init__(self) -> None:
        if self.fold_count < 2:
            raise ValueError("Joint span transformer OOF requires at least two folds")
        if not self.inference_device.strip():
            raise ValueError("Joint span transformer OOF inference_device is required")


@dataclass(frozen=True, slots=True)
class _OofExample:
    """Validated dataset row carrying exactly the data needed for fold routing and scoring."""

    row: Mapping[str, Any]
    candidate: Phase1JointSpanCandidate
    label: Phase1JointSpanLabel
    source_dataset: str


def assign_phase1_joint_span_oof_folds(
    document_ids: Sequence[str],
    *,
    fold_count: int,
) -> dict[str, int]:
    """Assign documents once using a deterministic hash order and balanced round robin.

    SCALING: grouping is by whole document, so every lattice alternative from a note is held out
    together. A hash-sorted round robin avoids empty folds without depending on filesystem order.
    """

    unique = sorted(set(document_ids))
    if len(unique) != len(document_ids):
        raise ValueError("Joint span OOF document IDs must be unique")
    if len(unique) < fold_count:
        raise ValueError("Joint span transformer OOF needs at least one document per fold")
    ordered = sorted(
        unique,
        key=lambda document_id: (
            hashlib.sha256(document_id.encode("utf-8")).hexdigest(),
            document_id,
        ),
    )
    return {document_id: index % fold_count for index, document_id in enumerate(ordered)}


def run_phase1_joint_span_transformer_oof(
    config: Phase1JointSpanOofConfig,
) -> Mapping[str, Any]:
    """Cross-fit local transformer verifiers and write calibration-ready OOF observations.

    The output is diagnostic only. It is a required input to a later calibration artifact, while
    official BTC submissions remain the only promotion evidence for model quality.
    """

    summary = inspect_phase1_joint_span_training_inputs(config.training)
    examples = _load_examples(config.training.dataset_path)
    document_ids = sorted({example.candidate.variant.document_id for example in examples})
    fold_by_document = assign_phase1_joint_span_oof_folds(
        document_ids,
        fold_count=config.fold_count,
    )
    _validate_fold_training_support(examples, fold_by_document, config.fold_count)
    family_fingerprint = phase1_joint_span_training_family_fingerprint(
        config.training,
        summary,
    )
    _prepare_output_directory(config)
    assignment_payload = _fold_assignment_payload(
        fold_by_document,
        dataset_sha256=summary.dataset_sha256,
        training_family_fingerprint=family_fingerprint,
    )
    assignment_path = config.output_dir / "fold_assignments.json"
    assignment_sha256 = write_json(assignment_path, assignment_payload)
    observations: list[Phase1JointSpanCalibrationObservation] = []
    fold_reports: list[dict[str, Any]] = []
    for fold in range(config.fold_count):
        fold_examples = tuple(
            example
            for example in examples
            if fold_by_document[example.candidate.variant.document_id] == fold
        )
        train_examples = tuple(
            example
            for example in examples
            if fold_by_document[example.candidate.variant.document_id] != fold
        )
        if not fold_examples or not train_examples:
            raise ValueError(f"Joint span OOF fold {fold} has empty train or validation data")
        fold_directory = config.output_dir / "folds" / f"fold-{fold}"
        fold_observations, fold_report = _run_or_resume_fold(
            config,
            fold=fold,
            fold_directory=fold_directory,
            train_examples=train_examples,
            validation_examples=fold_examples,
            full_dataset_sha256=summary.dataset_sha256,
            training_family_fingerprint=family_fingerprint,
        )
        observations.extend(fold_observations)
        fold_reports.append(fold_report)
    ordered_observations = tuple(
        sorted(
            observations,
            key=lambda item: (item.document_id, item.variant_id, item.fold),
        )
    )
    coverage = _validate_oof_coverage(examples, ordered_observations, fold_by_document)
    observations_path = config.output_dir / "oof_observations.jsonl"
    observations_sha256 = write_jsonl(
        observations_path,
        (observation.to_dict() for observation in ordered_observations),
    )
    manifest = {
        "schema_version": _OOF_SCHEMA,
        "purpose": "document_grouped_transformer_oof_calibration_only",
        "promotion": "official_submission_metrics_only",
        "training_family_fingerprint": family_fingerprint,
        "input": {
            "dataset": str(config.training.dataset_path),
            "dataset_sha256": summary.dataset_sha256,
            "dataset_manifest_sha256": sha256_file(config.training.dataset_manifest_path),
            "example_count": summary.example_count,
            "document_count": summary.document_count,
        },
        "fold_assignment": {
            "path": str(assignment_path),
            "sha256": assignment_sha256,
            "fold_count": config.fold_count,
        },
        "observations": {
            "path": str(observations_path),
            "sha256": observations_sha256,
            "count": len(ordered_observations),
        },
        "coverage": coverage,
        "folds": fold_reports,
        "policy": {
            "raw_text_in_observations": False,
            "round2_included": False,
            "friend31_included": False,
            "document_grouped": True,
        },
    }
    write_json(config.output_dir / "manifest.json", manifest)
    return manifest


def _run_or_resume_fold(
    config: Phase1JointSpanOofConfig,
    *,
    fold: int,
    fold_directory: Path,
    train_examples: Sequence[_OofExample],
    validation_examples: Sequence[_OofExample],
    full_dataset_sha256: str,
    training_family_fingerprint: str,
) -> tuple[tuple[Phase1JointSpanCalibrationObservation, ...], dict[str, Any]]:
    """Train or reuse a complete fold without accepting partial checkpoint state."""

    observations_path = fold_directory / "validation_observations.jsonl"
    fold_manifest_path = fold_directory / "fold_manifest.json"
    if config.resume and observations_path.is_file() and fold_manifest_path.is_file():
        return _load_completed_fold(
            fold_manifest_path,
            observations_path,
            fold=fold,
            validation_examples=validation_examples,
            training_family_fingerprint=training_family_fingerprint,
        )
    if fold_directory.exists() and any(fold_directory.iterdir()):
        raise ValueError(
            f"Joint span OOF fold output already exists: {fold_directory}; use --resume only "
            "after a complete fold manifest and validation observations were written"
        )
    dataset_directory = fold_directory / "dataset"
    model_directory = fold_directory / "model"
    fold_directory.mkdir(parents=True, exist_ok=True)
    dataset_path, dataset_manifest_path, dataset_sha256 = _write_fold_dataset(
        dataset_directory,
        train_examples,
        fold=fold,
        full_dataset_sha256=full_dataset_sha256,
    )
    fold_training = replace(
        config.training,
        dataset_path=dataset_path,
        dataset_manifest_path=dataset_manifest_path,
        output_dir=model_directory,
        training_family_dataset_sha256=full_dataset_sha256,
        overwrite_output=False,
    )
    training_manifest = train_phase1_joint_span_verifier(fold_training)
    verification = verify_phase1_joint_span_verifier_artifact(fold_training)
    if training_manifest.get("training_family_fingerprint") != training_family_fingerprint:
        raise RuntimeError("Joint span OOF fold training family drifted from its root dataset")
    model = training_manifest.get("model")
    if not isinstance(model, Mapping) or not isinstance(model.get("fingerprint"), str):
        raise RuntimeError("Joint span OOF fold training did not report a model fingerprint")
    verifier = HuggingFacePhase1JointSpanVerifier(
        HuggingFaceModelConfig(
            model_id=str(model_directory / "final-model"),
            revision=str(model["fingerprint"]),
            device=config.inference_device,
            batch_size=config.training.evaluation_batch_size,
            max_length=config.training.max_length,
        )
    )
    predictions = verifier.predict(tuple(item.candidate for item in validation_examples))
    if len(predictions) != len(validation_examples):
        raise RuntimeError("Joint span OOF fold verifier did not score every validation candidate")
    observations = tuple(
        Phase1JointSpanCalibrationObservation(
            document_id=example.candidate.variant.document_id,
            variant_id=example.candidate.variant.variant_id,
            fold=f"fold-{fold}",
            genre=example.candidate.genre,
            entity_type=example.candidate.variant.entity_type,
            exact_probability=prediction.exact_probability(example.candidate),
            is_exact=example.label is example.candidate.expected_exact_label,
        )
        for example, prediction in zip(validation_examples, predictions, strict=True)
    )
    observations_sha256 = write_jsonl(
        observations_path,
        (observation.to_dict() for observation in observations),
    )
    validation_document_count = len(
        {item.candidate.variant.document_id for item in validation_examples}
    )
    fold_manifest = {
        "schema_version": "phase1-joint-span-transformer-oof-fold.v1",
        "fold": fold,
        "training_family_fingerprint": training_family_fingerprint,
        "training_dataset_sha256": dataset_sha256,
        "training": training_manifest,
        "verification": verification,
        "validation": {
            "document_count": validation_document_count,
            "candidate_count": len(validation_examples),
            "observations_sha256": observations_sha256,
        },
    }
    write_json(fold_manifest_path, fold_manifest)
    return observations, {
        "fold": fold,
        "status": "trained",
        "training_document_count": len(
            {item.candidate.variant.document_id for item in train_examples}
        ),
        "validation_document_count": validation_document_count,
        "training_candidate_count": len(train_examples),
        "validation_candidate_count": len(validation_examples),
        "model_fingerprint": model["fingerprint"],
    }


def _load_completed_fold(
    manifest_path: Path,
    observations_path: Path,
    *,
    fold: int,
    validation_examples: Sequence[_OofExample],
    training_family_fingerprint: str,
) -> tuple[tuple[Phase1JointSpanCalibrationObservation, ...], dict[str, Any]]:
    """Reuse only a verified whole fold; incomplete state is never silently continued."""

    payload = _load_mapping(manifest_path, "joint span OOF fold manifest")
    if payload.get("schema_version") != "phase1-joint-span-transformer-oof-fold.v1":
        raise ValueError("Unsupported joint span OOF fold manifest schema")
    if payload.get("fold") != fold:
        raise ValueError("Joint span OOF fold manifest has a mismatched fold number")
    if payload.get("training_family_fingerprint") != training_family_fingerprint:
        raise ValueError("Joint span OOF resumed fold has a different training family")
    validation = _mapping(payload.get("validation"), "joint span OOF fold validation")
    if validation.get("observations_sha256") != sha256_file(observations_path):
        raise ValueError("Joint span OOF resumed observations changed after training")
    observations = _load_observations(observations_path)
    expected_ids = {item.candidate.variant.variant_id for item in validation_examples}
    if {item.variant_id for item in observations} != expected_ids:
        raise ValueError("Joint span OOF resumed observations do not cover the validation fold")
    if any(item.fold != f"fold-{fold}" for item in observations):
        raise ValueError("Joint span OOF resumed observations have the wrong fold identity")
    return observations, {
        "fold": fold,
        "status": "resumed",
        "training_document_count": None,
        "validation_document_count": len(
            {item.candidate.variant.document_id for item in validation_examples}
        ),
        "training_candidate_count": None,
        "validation_candidate_count": len(validation_examples),
        "model_fingerprint": _model_fingerprint_from_fold_manifest(payload),
    }


def _write_fold_dataset(
    output: Path,
    examples: Sequence[_OofExample],
    *,
    fold: int,
    full_dataset_sha256: str,
) -> tuple[Path, Path, str]:
    """Materialize one train-only fold subset in the normal training-data contract."""

    output.mkdir(parents=True, exist_ok=True)
    path = output / "examples.jsonl"
    manifest_path = output / "manifest.json"
    rows = tuple(item.row for item in examples)
    dataset_sha256 = write_jsonl(path, rows)
    write_json(
        manifest_path,
        {
            "schema_version": _FOLD_DATASET_SCHEMA,
            "examples_sha256": dataset_sha256,
            "example_count": len(rows),
            "source_document_count": len(
                {item.candidate.variant.document_id for item in examples}
            ),
            "label_counts": dict(
                sorted(Counter(item.label.value for item in examples).items())
            ),
            "oof": {
                "fold": fold,
                "full_dataset_sha256": full_dataset_sha256,
                "role": "training_only",
            },
        },
    )
    return path, manifest_path, dataset_sha256


def _load_examples(path: Path) -> tuple[_OofExample, ...]:
    """Parse model-ready rows and preserve their canonical serialized form for fold datasets."""

    rows: list[_OofExample] = []
    seen_variant_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid joint span JSON") from error
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{line_number}: joint span example must be an object")
            candidate = _candidate_from_row(row)
            if candidate.variant.variant_id in seen_variant_ids:
                raise ValueError(f"{path}:{line_number}: duplicate joint span variant ID")
            seen_variant_ids.add(candidate.variant.variant_id)
            label = _label_from_row(row)
            source_dataset = row.get("source_dataset")
            if not isinstance(source_dataset, str) or not source_dataset.strip():
                raise ValueError(f"{path}:{line_number}: source_dataset is required")
            rows.append(
                _OofExample(
                    row=dict(row),
                    candidate=candidate,
                    label=label,
                    source_dataset=source_dataset,
                )
            )
    if not rows:
        raise ValueError("Joint span transformer OOF dataset is empty")
    return tuple(sorted(rows, key=lambda item: item.candidate.variant.variant_id))


def _candidate_from_row(row: Mapping[str, Any]) -> Phase1JointSpanCandidate:
    variant = Phase1BoundaryVariant.from_dict(row)
    genre = row.get("genre")
    section = row.get("section")
    cross_encoder_text = row.get("cross_encoder_text")
    if not isinstance(genre, str) or not isinstance(section, str) or not isinstance(cross_encoder_text, str):
        raise ValueError("Joint span OOF example lacks genre, section, or cross_encoder_text")
    return Phase1JointSpanCandidate(
        variant=variant,
        genre=Phase1GenreBucket(genre).value,
        section=section,
        cross_encoder_text=cross_encoder_text,
    )


def _label_from_row(row: Mapping[str, Any]) -> Phase1JointSpanLabel:
    value = row.get("label")
    if not isinstance(value, str):
        raise ValueError("Joint span OOF example label is required")
    try:
        return Phase1JointSpanLabel(value)
    except ValueError as error:
        raise ValueError("Joint span OOF example label is unsupported") from error


def _validate_fold_training_support(
    examples: Sequence[_OofExample],
    fold_by_document: Mapping[str, int],
    fold_count: int,
) -> None:
    """Reject folds whose train subset cannot satisfy the eight-class model contract."""

    expected_labels = set(_LABELS)
    for fold in range(fold_count):
        observed = {
            item.label.value
            for item in examples
            if fold_by_document[item.candidate.variant.document_id] != fold
        }
        missing = sorted(expected_labels - observed)
        if missing:
            raise ValueError(
                f"Joint span OOF fold {fold} training split lacks labels: {missing}"
            )


def _validate_oof_coverage(
    examples: Sequence[_OofExample],
    observations: Sequence[Phase1JointSpanCalibrationObservation],
    fold_by_document: Mapping[str, int],
) -> dict[str, Any]:
    """Prove one OOF prediction per lattice candidate and report its real denominator."""

    expected_by_id = {item.candidate.variant.variant_id: item for item in examples}
    observed_by_id = {item.variant_id: item for item in observations}
    if len(observed_by_id) != len(observations):
        raise ValueError("Joint span transformer OOF observations contain duplicate variant IDs")
    if set(expected_by_id) != set(observed_by_id):
        missing = sorted(set(expected_by_id) - set(observed_by_id))
        extra = sorted(set(observed_by_id) - set(expected_by_id))
        raise ValueError(
            "Joint span transformer OOF coverage mismatch: "
            f"missing={len(missing)} extra={len(extra)}"
        )
    for variant_id, example in expected_by_id.items():
        observation = observed_by_id[variant_id]
        document_id = example.candidate.variant.document_id
        if observation.document_id != document_id:
            raise ValueError("Joint span transformer OOF observation document changed")
        if observation.fold != f"fold-{fold_by_document[document_id]}":
            raise ValueError("Joint span transformer OOF observation used the wrong fold")
        if observation.genre != example.candidate.genre or observation.entity_type != example.candidate.variant.entity_type:
            raise ValueError("Joint span transformer OOF observation changed candidate type evidence")
        if observation.is_exact != (example.label is example.candidate.expected_exact_label):
            raise ValueError("Joint span transformer OOF observation changed its gold target")
    expected_documents = set(fold_by_document)
    observed_documents = {item.document_id for item in observations}
    if observed_documents != expected_documents:
        raise ValueError("Joint span transformer OOF did not cover every source document")
    by_genre_type: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"expected": 0, "scored": 0, "positive": 0, "negative": 0}
    )
    for example in examples:
        key = (example.candidate.genre, example.candidate.variant.entity_type)
        by_genre_type[key]["expected"] += 1
        if example.label is example.candidate.expected_exact_label:
            by_genre_type[key]["positive"] += 1
        else:
            by_genre_type[key]["negative"] += 1
    for observation in observations:
        by_genre_type[(observation.genre, observation.entity_type)]["scored"] += 1
    return {
        "expected_candidate_count": len(examples),
        "scored_candidate_count": len(observations),
        "candidate_coverage": len(observations) / len(examples),
        "expected_document_count": len(expected_documents),
        "scored_document_count": len(observed_documents),
        "document_coverage": len(observed_documents) / len(expected_documents),
        "by_genre_type": [
            {"genre": genre, "type": entity_type, **counts}
            for (genre, entity_type), counts in sorted(by_genre_type.items())
        ],
    }


def _fold_assignment_payload(
    fold_by_document: Mapping[str, int],
    *,
    dataset_sha256: str,
    training_family_fingerprint: str,
) -> dict[str, Any]:
    """Write a separate human-readable assignment before GPU work starts."""

    assignments = [
        {"document_id": document_id, "fold": fold}
        for document_id, fold in sorted(fold_by_document.items())
    ]
    return {
        "schema_version": _FOLD_ASSIGNMENT_SCHEMA,
        "dataset_sha256": dataset_sha256,
        "training_family_fingerprint": training_family_fingerprint,
        "fold_count": max(fold_by_document.values()) + 1,
        "assignments": assignments,
    }


def _load_observations(path: Path) -> tuple[Phase1JointSpanCalibrationObservation, ...]:
    rows: list[Phase1JointSpanCalibrationObservation] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid OOF observation JSON") from error
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_number}: OOF observation must be an object")
            rows.append(Phase1JointSpanCalibrationObservation.from_dict(payload))
    return tuple(rows)


def _model_fingerprint_from_fold_manifest(payload: Mapping[str, Any]) -> str:
    training = _mapping(payload.get("training"), "joint span OOF resumed training")
    model = _mapping(training.get("model"), "joint span OOF resumed model")
    fingerprint = model.get("fingerprint")
    if not isinstance(fingerprint, str):
        raise ValueError("Joint span OOF resumed model fingerprint is missing")
    return fingerprint


def _prepare_output_directory(config: Phase1JointSpanOofConfig) -> None:
    """Create a fresh OOF root, or only resume a root that already has a manifest scaffold."""

    if config.output_dir.exists() and any(config.output_dir.iterdir()) and not config.resume:
        raise ValueError("Joint span transformer OOF output exists; pass --resume to reuse folds")
    config.output_dir.mkdir(parents=True, exist_ok=True)


def _load_mapping(path: Path, name: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be an object")
    return payload


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    """Narrow an untrusted JSON field to a mapping with an actionable error."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value
