"""Labeled proposal data for diagnostic calibration and governed final fitting.

This builder consumes an unlabeled proposal matrix but reads labels only for the frozen model
train/development ids by default. A strict training-governance file may explicitly supersede the
legacy split and authorize all reviewed labels for final fitting.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clingrounder.benchmarks.phase1.proposal_features import (
    PHASE1_PROPOSAL_FEATURE_CONTRACT,
    ProposalSourceRole,
    extract_phase1_proposal_context,
    extract_phase1_proposal_features,
    phase1_genre_bucket,
)
from clingrounder.benchmarks.phase1.training_governance import (
    load_phase1_training_governance,
)
from clingrounder.ner.document_structure import DocumentStructureAnalyzer
from clingrounder.utils.hashing import sha256_file

__all__ = [
    "Phase1ProposalDataset",
    "Phase1ProposalExample",
    "build_phase1_proposal_dataset",
    "write_phase1_proposal_dataset",
]

_DATASET_SCHEMA = "phase1-proposal-calibration-dataset.v4"
_ERROR_KINDS = frozenset(
    {"exact", "boundary", "type_confusion", "boundary_type_confusion", "spurious"}
)


@dataclass(frozen=True, slots=True)
class Phase1ProposalExample:
    """One proposal label and sparse features for a frozen model split."""

    document_id: str
    proposal_id: str
    split: str
    text: str
    entity_type: str
    position: tuple[int, int]
    sources: tuple[str, ...]
    status: str
    label: int
    error_kind: str
    features: tuple[tuple[str, float], ...]
    left_context: str = ""
    right_context: str = ""
    section: str = "none"
    genre: str = "unknown"
    question_answer_role: str = "none"
    source_evidence: tuple[
        tuple[str, float | None, tuple[str, ...], bool],
        ...,
    ] = ()

    def __post_init__(self) -> None:
        if self.split not in {"train", "development"}:
            raise ValueError("Proposal example must belong to train or development")
        if self.label not in {0, 1}:
            raise ValueError("Proposal label must be binary")
        if self.error_kind not in _ERROR_KINDS:
            raise ValueError(f"Unsupported proposal error kind {self.error_kind!r}")
        if (self.label == 1) != (self.error_kind == "exact"):
            raise ValueError("Only exact span/type proposals may have a positive label")

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "proposal_id": self.proposal_id,
            "split": self.split,
            "text": self.text,
            "type": self.entity_type,
            "position": list(self.position),
            "sources": list(self.sources),
            "status": self.status,
            "label": self.label,
            "error_kind": self.error_kind,
            "left_context": self.left_context,
            "right_context": self.right_context,
            "section": self.section,
            "genre": self.genre,
            "question_answer_role": self.question_answer_role,
            "source_evidence": {
                source: {
                    "present": True,
                    "confidence": confidence,
                    "source_labels": list(source_labels),
                    "support_only": support_only,
                }
                for source, confidence, source_labels, support_only in self.source_evidence
            },
            "features": dict(self.features),
        }


@dataclass(frozen=True, slots=True)
class Phase1ProposalDataset:
    """In-memory proposal examples plus their immutable build manifest."""

    examples: tuple[Phase1ProposalExample, ...]
    manifest: Mapping[str, Any]


def build_phase1_proposal_dataset(
    matrix_path: str | Path,
    input_dir: str | Path,
    gold_dir: str | Path,
    model_split_manifest_path: str | Path,
    frozen_holdout_manifest_path: str | Path,
    *,
    source_roles: Mapping[str, ProposalSourceRole | str],
    training_governance_path: str | Path | None = None,
) -> Phase1ProposalDataset:
    """Build exact proposal labels under either the legacy or final-fit policy.

    INVARIANT: only ids listed under ``source_document_ids.train`` and ``development`` are read
    by default. Passing final-fit governance explicitly opens all 100 reviewed labels for
    supervised fitting; Round 2 remains excluded.
    """

    matrix_file = Path(matrix_path)
    input_root = Path(input_dir)
    gold_root = Path(gold_dir)
    model_manifest_file = Path(model_split_manifest_path)
    holdout_manifest_file = Path(frozen_holdout_manifest_path)
    model_manifest = _read_mapping(model_manifest_file)
    holdout_manifest = _read_mapping(holdout_manifest_file)
    split_by_document, holdout_ids = _validate_split_contracts(
        model_manifest,
        holdout_manifest,
    )
    selected_ids = frozenset(split_by_document)
    holdout_labels_read = False
    governance_file: Path | None = None
    if training_governance_path is not None:
        governance_file = Path(training_governance_path)
        governance = load_phase1_training_governance(governance_file)
        if governance.can_local_metric_decide():
            raise ValueError("Final-fit governance cannot authorize local promotion")
        selected_ids = frozenset({*selected_ids, *holdout_ids})
        if len(selected_ids) != governance.manual_gold.expected_document_count:
            raise ValueError(
                "Final-fit manual-gold document count does not match governance"
            )
        split_by_document.update(
            {document_id: "train" for document_id in holdout_ids}
        )
        holdout_labels_read = True
    assignments = _assignment_by_document(holdout_manifest)

    documents: dict[str, str] = {}
    gold_by_document: dict[str, list[dict[str, Any]]] = {}
    for document_id in sorted(selected_ids, key=_document_sort_key):
        document_path = input_root / f"{document_id}.txt"
        gold_path = gold_root / f"{document_id}.json"
        assignment = assignments.get(document_id)
        if assignment is None:
            raise ValueError(f"Frozen manifest has no assignment for document {document_id}")
        _verify_file_hash(document_path, str(assignment.get("document_sha256", "")))
        _verify_file_hash(gold_path, str(assignment.get("gold_sha256", "")))
        documents[document_id] = document_path.read_text(encoding="utf-8")
        gold_by_document[document_id] = _read_gold_rows(gold_path)

    rows = _read_jsonl(matrix_file)
    examples: list[Phase1ProposalExample] = []
    analyzer = DocumentStructureAnalyzer()
    structures = {
        document_id: analyzer.analyze(source_text)
        for document_id, source_text in documents.items()
    }
    for row in rows:
        document_id = str(row.get("document_id", ""))
        if (
            not holdout_labels_read
            and document_id in holdout_ids
        ) or document_id not in selected_ids:
            continue
        source_text = documents[document_id]
        position = _position(row)
        entity_type = str(row.get("type", ""))
        text = str(row.get("text", ""))
        if source_text[position[0] : position[1]] != text:
            raise ValueError(
                f"Proposal {row.get('proposal_id')} no longer matches document {document_id}"
            )
        label, error_kind = _proposal_label(
            position,
            entity_type,
            gold_by_document[document_id],
        )
        features = extract_phase1_proposal_features(
            row,
            source_text,
            source_roles,
            structure=structures[document_id],
        )
        context = extract_phase1_proposal_context(
            row,
            source_text,
            structure=structures[document_id],
        )
        raw_sources = row.get("sources")
        assert isinstance(raw_sources, list)
        sources = tuple(sorted(str(source) for source in raw_sources))
        examples.append(
            Phase1ProposalExample(
                document_id=document_id,
                proposal_id=str(row.get("proposal_id", "")),
                split=split_by_document[document_id],
                text=text,
                entity_type=entity_type,
                position=position,
                sources=sources,
                status=str(row.get("status", "unknown")),
                label=label,
                error_kind=error_kind,
                features=tuple(sorted(features.items())),
                left_context=context.left_context,
                right_context=context.right_context,
                section=context.section,
                genre=context.genre,
                question_answer_role=context.question_answer_role,
                source_evidence=_source_evidence(row, sources),
            )
        )
    examples.sort(key=_example_sort_key)
    manifest = _build_manifest(
        examples,
        gold_by_document,
        split_by_document,
        matrix_file,
        model_manifest_file,
        holdout_manifest_file,
        model_manifest,
        holdout_manifest,
        source_roles,
        {
            document_id: phase1_genre_bucket(structure.genre).value
            for document_id, structure in structures.items()
        },
        holdout_labels_read=holdout_labels_read,
        training_governance_path=governance_file,
    )
    return Phase1ProposalDataset(examples=tuple(examples), manifest=manifest)


def write_phase1_proposal_dataset(
    dataset: Phase1ProposalDataset,
    output_dir: str | Path,
) -> None:
    """Write deterministic, inspectable JSON artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(dataset.manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "examples.jsonl").write_text(
        _serialized_examples(dataset.examples),
        encoding="utf-8",
    )


def _validate_split_contracts(
    model_manifest: Mapping[str, Any],
    holdout_manifest: Mapping[str, Any],
) -> tuple[dict[str, str], frozenset[str]]:
    if model_manifest.get("schema_version") != "phase1-model-training-split.v1":
        raise ValueError("Unsupported Phase 1 model split manifest")
    if bool(model_manifest.get("round2_included", True)):
        raise ValueError("Round 2 cannot enter proposal calibration")
    if holdout_manifest.get("schema_version") != "phase1-manual-gold-split.v1":
        raise ValueError("Unsupported frozen holdout manifest")

    source_ids = model_manifest.get("source_document_ids")
    frozen_splits = holdout_manifest.get("splits")
    if not isinstance(source_ids, Mapping) or not isinstance(frozen_splits, Mapping):
        raise ValueError("Split manifests are missing document ids")
    train_ids = _string_ids(source_ids.get("train"), "model train")
    development_ids = _string_ids(source_ids.get("development"), "model development")
    holdout_section = frozen_splits.get("holdout")
    source_train_section = frozen_splits.get("train")
    if not isinstance(holdout_section, Mapping) or not isinstance(
        source_train_section, Mapping
    ):
        raise ValueError("Frozen split manifest is incomplete")
    holdout_ids = _string_ids(holdout_section.get("document_ids"), "frozen holdout")
    source_train_ids = _string_ids(
        source_train_section.get("document_ids"),
        "frozen source train",
    )
    if train_ids & development_ids:
        raise ValueError("Model train and development ids overlap")
    if (train_ids | development_ids) != source_train_ids:
        raise ValueError("Model train/development ids do not partition frozen source train")
    if (train_ids | development_ids) & holdout_ids:
        raise ValueError("Model calibration ids overlap frozen holdout")

    excluded = model_manifest.get("excluded_holdout")
    if not isinstance(excluded, Mapping):
        raise ValueError("Model split does not fingerprint excluded holdout")
    if excluded.get("document_ids_sha256") != _ids_sha256(holdout_ids):
        raise ValueError("Model split excluded-holdout fingerprint is stale")
    model_corpus = model_manifest.get("source_corpus_fingerprint_sha256")
    frozen_corpus = holdout_manifest.get("corpus")
    if not isinstance(frozen_corpus, Mapping) or model_corpus != frozen_corpus.get(
        "fingerprint_sha256"
    ):
        raise ValueError("Model and frozen holdout manifests describe different corpora")

    split_by_document = {document_id: "train" for document_id in train_ids}
    split_by_document.update(
        {document_id: "development" for document_id in development_ids}
    )
    return split_by_document, frozenset(holdout_ids)


def _proposal_label(
    position: tuple[int, int],
    entity_type: str,
    gold_rows: Sequence[Mapping[str, Any]],
) -> tuple[int, str]:
    start, end = position
    same_span = False
    overlapping_same_type = False
    overlapping_other_type = False
    for gold in gold_rows:
        gold_position = _position(gold)
        gold_type = str(gold.get("type", ""))
        if gold_position == position:
            if gold_type == entity_type:
                return 1, "exact"
            same_span = True
            continue
        overlap = min(end, gold_position[1]) > max(start, gold_position[0])
        if not overlap:
            continue
        if gold_type == entity_type:
            overlapping_same_type = True
        else:
            overlapping_other_type = True
    if same_span:
        return 0, "type_confusion"
    if overlapping_same_type:
        return 0, "boundary"
    if overlapping_other_type:
        return 0, "boundary_type_confusion"
    return 0, "spurious"


def _build_manifest(
    examples: Sequence[Phase1ProposalExample],
    gold_by_document: Mapping[str, Sequence[Mapping[str, Any]]],
    split_by_document: Mapping[str, str],
    matrix_path: Path,
    model_manifest_path: Path,
    holdout_manifest_path: Path,
    model_manifest: Mapping[str, Any],
    holdout_manifest: Mapping[str, Any],
    source_roles: Mapping[str, ProposalSourceRole | str],
    genre_by_document: Mapping[str, str],
    *,
    holdout_labels_read: bool,
    training_governance_path: Path | None,
) -> dict[str, Any]:
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    gold_counts: Counter[str] = Counter()
    gold_genre_counts: Counter[str] = Counter()
    for example in examples:
        split_counts[example.split] += 1
        label_counts[f"{example.split}:{example.label}"] += 1
        type_counts[f"{example.split}:{example.entity_type}:{example.label}"] += 1
        status_counts[f"{example.split}:{example.status}:{example.label}"] += 1
        error_counts[f"{example.split}:{example.error_kind}"] += 1
    for document_id, gold_rows in gold_by_document.items():
        split = split_by_document[document_id]
        genre = genre_by_document.get(document_id, "unknown")
        for row in gold_rows:
            entity_type = str(row.get("type", ""))
            gold_counts[f"{split}:{entity_type}"] += 1
            gold_genre_counts[f"{split}:{genre}:{entity_type}"] += 1
    frozen_corpus = holdout_manifest.get("corpus")
    assert isinstance(frozen_corpus, Mapping)
    return {
        "schema_version": _DATASET_SCHEMA,
        "feature_contract": PHASE1_PROPOSAL_FEATURE_CONTRACT,
        "example_count": len(examples),
        "examples_sha256": hashlib.sha256(
            _serialized_examples(examples).encode("utf-8")
        ).hexdigest(),
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "type_label_counts": dict(sorted(type_counts.items())),
        "status_label_counts": dict(sorted(status_counts.items())),
        "error_counts": dict(sorted(error_counts.items())),
        "gold_entity_counts": dict(sorted(gold_counts.items())),
        "gold_entity_genre_counts": dict(sorted(gold_genre_counts.items())),
        "document_genres": dict(
            sorted(genre_by_document.items(), key=lambda item: _document_sort_key(item[0]))
        ),
        "source_roles": {
            source: ProposalSourceRole(role).value
            for source, role in sorted(source_roles.items())
        },
        "inputs": {
            "proposal_matrix_sha256": sha256_file(matrix_path),
            "model_split_manifest_sha256": sha256_file(model_manifest_path),
            "frozen_holdout_manifest_sha256": sha256_file(holdout_manifest_path),
            "source_corpus_fingerprint_sha256": frozen_corpus.get(
                "fingerprint_sha256"
            ),
            "round2_included": False,
            "holdout_labels_read": holdout_labels_read,
            "excluded_holdout_document_ids_sha256": model_manifest[
                "excluded_holdout"
            ]["document_ids_sha256"],
            "training_governance_sha256": (
                sha256_file(training_governance_path)
                if training_governance_path is not None
                else None
            ),
        },
        "decision_authority": {
            "local_metrics": "diagnostic_only",
            "auto_promote": False,
            "official_submission_required": True,
        },
        "label_policy": {
            "positive": "exact_raw_span_and_exact_phase1_type",
            "negative": [
                "boundary",
                "type_confusion",
                "boundary_type_confusion",
                "spurious",
            ],
        },
    }


def _source_evidence(
    row: Mapping[str, Any],
    sources: tuple[str, ...],
) -> tuple[tuple[str, float | None, tuple[str, ...], bool], ...]:
    raw = row.get("source_evidence", {})
    if not isinstance(raw, Mapping):
        raise ValueError("Proposal source_evidence must be an object")
    values: list[tuple[str, float | None, tuple[str, ...], bool]] = []
    for source in sources:
        evidence = raw.get(source, {})
        if not isinstance(evidence, Mapping):
            raise ValueError(f"Proposal evidence for {source!r} must be an object")
        confidence = evidence.get("confidence")
        if confidence is not None and (
            not isinstance(confidence, int | float) or isinstance(confidence, bool)
        ):
            raise ValueError(f"Proposal confidence for {source!r} must be numeric")
        labels = evidence.get("source_labels", [])
        if not isinstance(labels, list) or not all(
            isinstance(label, str) and label for label in labels
        ):
            raise ValueError(f"Proposal labels for {source!r} must be strings")
        values.append(
            (
                source,
                float(confidence) if confidence is not None else None,
                tuple(sorted(set(labels))),
                bool(evidence.get("support_only", False)),
            )
        )
    return tuple(values)


def _serialized_examples(examples: Sequence[Phase1ProposalExample]) -> str:
    """Serialize feature rows exactly once for both hashing and persisted output."""

    return "".join(
        json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for example in examples
    )


def _assignment_by_document(
    holdout_manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    raw = holdout_manifest.get("assignments")
    if not isinstance(raw, list):
        raise ValueError("Frozen holdout manifest has no assignments")
    assignments: dict[str, Mapping[str, Any]] = {}
    for row in raw:
        if not isinstance(row, Mapping):
            raise ValueError("Frozen assignment must be an object")
        document_id = str(row.get("document_id", ""))
        if not document_id or document_id in assignments:
            raise ValueError("Frozen assignments contain a missing or duplicate document id")
        assignments[document_id] = row
    return assignments


def _verify_file_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if not expected or actual != expected:
        raise ValueError(f"Frozen source fingerprint mismatch for {path}")


def _read_gold_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"{path}: expected a JSON entity list")
    return payload


def _read_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(payload)
    return rows


def _position(row: Mapping[str, Any]) -> tuple[int, int]:
    value = row.get("position")
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        or value[0] >= value[1]
    ):
        raise ValueError("Proposal/gold row has an invalid position")
    return value[0], value[1]


def _string_ids(value: Any, name: str) -> set[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} document ids must be a string list")
    ids = set(value)
    if len(ids) != len(value):
        raise ValueError(f"{name} document ids contain duplicates")
    return ids


def _ids_sha256(values: Sequence[str] | set[str]) -> str:
    ordered = sorted(values, key=_document_sort_key)
    return hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()


def _example_sort_key(example: Phase1ProposalExample) -> tuple[Any, ...]:
    return (
        _document_sort_key(example.document_id),
        example.position[0],
        example.position[1],
        example.entity_type,
    )


def _document_sort_key(document_id: str) -> tuple[int, int | str]:
    return (0, int(document_id)) if document_id.isdigit() else (1, document_id)
