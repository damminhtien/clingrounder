"""Adapt the governed final token bundle into exact-span/type verifier supervision.

The token bundle is already the reproducible union of authorized clinical supervision and bounded
Q&A/educational augmentation.  Joint-span training needs the same immutable source strings, but
each token chunk becomes a separate document because its offsets are chunk-local.  OOF grouping
then keeps all chunks and synthetic children of one original note in a single fold.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.benchmarks.phase1.reviewed_corpus import Phase1ReviewedCorpus
from medical_kg_nlp.benchmarks.phase1.joint_span_preparation import (
    prepare_phase1_joint_span_supervision,
)
from medical_kg_nlp.benchmarks.phase1.proposal_features import ProposalSourceRole
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.benchmarks.phase1.ontology import PHASE1_TYPE_BY_ENTITY_TYPE
from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.training.span_dataset import (
    SpanTrainingRecord,
    scan_span_dataset,
    validate_span_dataset_manifest,
)
from medical_kg_nlp.utils.hashing import sha256_file

__all__ = [
    "Phase1JointSpanTokenBundle",
    "load_phase1_joint_span_token_bundle",
    "prepare_phase1_joint_span_token_bundle",
]

_SYNTHETIC_SOURCE = "synthetic:phase1-region-renderer.v1"
_FINAL_SOURCE_PREFIX = "phase1-final-supervision:"
_ALLOWED_NOTE_TYPES = frozenset(
    {"phase1_final_supervision", "question_answer", "educational"}
)


@dataclass(frozen=True, slots=True)
class Phase1JointSpanTokenBundle:
    """Joint-span corpus, source provenance, and grouped OOF contract for one bundle."""

    corpus: Phase1ReviewedCorpus
    source_dataset_by_document: Mapping[str, str]
    oof_group_by_document: Mapping[str, str]
    genre_by_document: Mapping[str, str]
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        document_ids = set(self.corpus.source_texts)
        if document_ids != set(self.corpus.gold_rows):
            raise ValueError("Joint span token bundle gold rows must cover every child document")
        if document_ids != set(self.corpus.split_by_document):
            raise ValueError("Joint span token bundle splits must cover every child document")
        if document_ids != set(self.source_dataset_by_document):
            raise ValueError("Joint span token bundle source provenance must cover every child document")
        if document_ids != set(self.oof_group_by_document):
            raise ValueError("Joint span token bundle OOF groups must cover every child document")
        if document_ids != set(self.genre_by_document):
            raise ValueError("Joint span token bundle genres must cover every child document")
        if set(self.corpus.split_by_document.values()) != {"train"}:
            raise ValueError("Joint span token bundle must be final-fit train supervision only")


def load_phase1_joint_span_token_bundle(
    *,
    dataset_path: str | Path,
    manifest_path: str | Path,
    build_manifest_path: str | Path | None = None,
) -> Phase1JointSpanTokenBundle:
    """Load a complete, provenance-checked token bundle as local joint-span documents.

    INVARIANT: ``record.text`` remains untouched.  Gold spans are copied only after
    ``SpanTrainingRecord`` validates that every local offset indexes that exact text.
    """

    dataset = Path(dataset_path)
    summary = scan_span_dataset(dataset)
    dataset_manifest = validate_span_dataset_manifest(dataset, manifest_path, summary)
    _validate_manifest_provenance(dataset_manifest)
    build_manifest = _load_build_manifest(build_manifest_path)
    if build_manifest is not None:
        _validate_build_manifest(build_manifest, dataset, summary.dataset_sha256)

    source_texts: dict[str, str] = {}
    gold_rows: dict[str, tuple[Mapping[str, Any], ...]] = {}
    source_dataset_by_document: dict[str, str] = {}
    oof_group_by_document: dict[str, str] = {}
    genre_by_document: dict[str, str] = {}
    record_count = 0
    for line_number, raw in _iter_raw_rows(dataset):
        try:
            record = SpanTrainingRecord.from_mapping(raw)
            _validate_record_provenance(raw, record)
        except ValueError as error:
            raise ValueError(f"{dataset}:{line_number}: {error}") from error
        child_document_id = f"joint-span-token:{record.record_id}"
        if child_document_id in source_texts:
            raise ValueError(f"Duplicate joint-span child document ID: {child_document_id}")
        source_texts[child_document_id] = record.text
        gold_rows[child_document_id] = tuple(
            _phase1_gold_row(entity, record=record) for entity in record.entities
        )
        source_dataset_by_document[child_document_id] = _source_dataset(raw, record)
        oof_group_by_document[child_document_id] = _oof_group(raw, record)
        genre_by_document[child_document_id] = _genre_for_record(record)
        record_count += 1
    if record_count != summary.record_count:
        raise RuntimeError("Joint span token bundle changed while it was being loaded")

    corpus = Phase1ReviewedCorpus(
        source_texts=source_texts,
        gold_rows=gold_rows,
        split_by_document={document_id: "train" for document_id in source_texts},
    )
    manifest: dict[str, Any] = {
        "schema_version": "phase1-joint-span-token-bundle.v1",
        "dataset": {
            "path": str(dataset),
            "sha256": summary.dataset_sha256,
            "manifest_sha256": sha256_file(manifest_path),
            "record_count": summary.record_count,
            "entity_count": summary.entity_count,
        },
        "child_document_count": len(source_texts),
        "oof_group_count": len(set(oof_group_by_document.values())),
        "genre_counts": _count_values(genre_by_document),
        "source_dataset_counts": _count_values(source_dataset_by_document),
        "round2_included": False,
        "friend31_included": False,
    }
    if build_manifest_path is not None:
        manifest["build_manifest"] = {
            "path": str(build_manifest_path),
            "sha256": sha256_file(build_manifest_path),
        }
    return Phase1JointSpanTokenBundle(
        corpus=corpus,
        source_dataset_by_document=source_dataset_by_document,
        oof_group_by_document=oof_group_by_document,
        genre_by_document=genre_by_document,
        manifest=manifest,
    )


def prepare_phase1_joint_span_token_bundle(
    bundle: Phase1JointSpanTokenBundle,
    dictionary: DictionaryStore,
    *,
    output_dir: str | Path,
    model_sources: Mapping[str, tuple[Path, ProposalSourceRole | str]] | None = None,
) -> dict[str, Any]:
    """Prepare mixed-genre joint-span supervision and preserve its grouped OOF contract.

    Bootstrap preparation intentionally permits only the two deterministic sources.  A later
    Qwen or XLM-R artifact must cover the exact same child document IDs and is pinned here before
    it can contribute to the production lattice.
    """

    return prepare_phase1_joint_span_supervision(
        bundle.corpus,
        dictionary,
        source_dataset_by_document=bundle.source_dataset_by_document,
        oof_group_by_document=bundle.oof_group_by_document,
        genre_by_document=bundle.genre_by_document,
        supervision_manifest=bundle.manifest,
        model_sources=model_sources or {},
        output_dir=output_dir,
        purpose="final_fit_authorized_supervision_with_bounded_qa_educational_augmentation",
        require_model_source=False,
    )


def _iter_raw_rows(path: Path) -> tuple[tuple[int, Mapping[str, Any]], ...]:
    """Read once so source metadata unavailable in ``SpanTrainingRecord`` stays governed."""

    rows: list[tuple[int, Mapping[str, Any]]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid span dataset JSON") from error
            if not isinstance(raw, Mapping):
                raise ValueError(f"{path}:{line_number}: span dataset row must be an object")
            rows.append((line_number, raw))
    return tuple(rows)


def _phase1_gold_row(entity: Any, *, record: SpanTrainingRecord) -> Mapping[str, Any]:
    """Translate one neutral five-type label without inventing Phase 1 metadata."""

    try:
        phase1_type = PHASE1_TYPE_BY_ENTITY_TYPE[EntityType(entity.label)]
    except (KeyError, ValueError) as error:
        raise ValueError(
            f"Joint span token record {record.record_id!r} has unsupported label {entity.label!r}"
        ) from error
    return {
        "text": entity.text,
        "type": phase1_type,
        "position": [entity.start, entity.end],
        "assertions": [],
        "candidates": [],
    }


def _validate_record_provenance(raw: Mapping[str, Any], record: SpanTrainingRecord) -> None:
    """Allow only the controlled final corpus and its bounded local augmentation."""

    if record.split != "train":
        raise ValueError("Joint span token records must use the final train split")
    if record.note_type not in _ALLOWED_NOTE_TYPES:
        raise ValueError(f"Joint span token record has an unsupported note type: {record.note_type}")
    source = record.source_artifact_id
    if source.startswith(_FINAL_SOURCE_PREFIX):
        if record.note_type != "phase1_final_supervision":
            raise ValueError("Final supervision source has a non-clinical note type")
        return
    if source != _SYNTHETIC_SOURCE:
        raise ValueError("Joint span token record has an unapproved source artifact")
    if record.note_type not in {"question_answer", "educational"}:
        raise ValueError("Synthetic joint span token record has an invalid note type")
    metadata = raw.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Synthetic joint span token record is missing provenance metadata")
    parent = metadata.get("parent_document_id")
    if not isinstance(parent, str) or not parent.strip():
        raise ValueError("Synthetic joint span token record is missing parent_document_id")


def _source_dataset(raw: Mapping[str, Any], record: SpanTrainingRecord) -> str:
    """Expose source class for later genre-aware calibration without leaking raw text."""

    if record.source_artifact_id == _SYNTHETIC_SOURCE:
        return f"synthetic_{record.note_type}"
    metadata = raw.get("metadata")
    if isinstance(metadata, Mapping):
        source_dataset = metadata.get("source_dataset")
        if isinstance(source_dataset, str) and source_dataset in {
            "manual_gold",
            "authorized_ground_truth",
        }:
            return source_dataset
    return "authorized_ground_truth" if record.document_id.startswith("authorized_gt:") else "manual_gold"


def _oof_group(raw: Mapping[str, Any], record: SpanTrainingRecord) -> str:
    """Group all chunks and derived children of one original note into one OOF fold."""

    metadata = raw.get("metadata")
    parent = metadata.get("parent_document_id") if isinstance(metadata, Mapping) else None
    origin = parent if isinstance(parent, str) and parent.strip() else record.document_id
    # Synthetic region rows inherit the exported manual dataset identity. The clinical token rows
    # use the bare Phase 1 document ID, so normalize this known provenance wrapper before grouping.
    if origin.startswith("phase1-manual-gold:"):
        origin = origin.removeprefix("phase1-manual-gold:")
    return f"phase1-origin:{origin}"


def _genre_for_record(record: SpanTrainingRecord) -> str:
    """Use the bundle's immutable note kind instead of reclassifying a renderer template."""

    return {
        "phase1_final_supervision": "clinical",
        "question_answer": "qa",
        "educational": "educational",
    }[record.note_type]


def _validate_manifest_provenance(manifest: Mapping[str, Any]) -> None:
    """Reject affirmative benchmark-prohibited provenance before source texts are loaded."""

    _reject_disallowed_provenance(manifest)
    if manifest.get("round2_included") is not False or manifest.get("friend31_included") is not False:
        raise ValueError("Joint span token bundle must explicitly exclude Round 2 and Friend31")


def _load_build_manifest(path: str | Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Joint span token bundle build manifest must be an object")
    _reject_disallowed_provenance(raw)
    return raw


def _validate_build_manifest(
    manifest: Mapping[str, Any],
    dataset_path: Path,
    dataset_sha256: str,
) -> None:
    """Bind an optional bundle build manifest to the exact dataset rather than its directory."""

    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("Joint span token bundle build manifest has no dataset object")
    if dataset.get("sha256") != dataset_sha256:
        raise ValueError("Joint span token bundle build manifest has a different dataset hash")
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if manifest.get("round2_included") is not False or manifest.get("friend31_included") is not False:
        raise ValueError("Joint span token bundle build manifest must exclude Round 2 and Friend31")


def _reject_disallowed_provenance(value: object, *, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            child_path = f"{path}.{key}"
            if (
                any(marker in key_text for marker in ("round2", "friend31", "quarantined"))
                and child is True
            ):
                raise ValueError(f"Joint span token bundle rejects disallowed provenance: {child_path}")
            _reject_disallowed_provenance(child, path=child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_disallowed_provenance(child, path=f"{path}[{index}]")


def _count_values(values: Mapping[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values.values():
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
