"""Strict JSON/JSONL serialization for resumable mining stages."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from medical_kg_nlp.mining.records import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    ConceptLink,
    MinedDocument,
    RedistributionPolicy,
    RelationProposal,
    ReviewStatus,
    SourceArtifact,
    StoredObject,
)

__all__ = [
    "annotation_from_dict",
    "document_from_dict",
    "load_annotations",
    "load_documents",
    "load_relations",
    "load_source_artifacts",
    "relation_from_dict",
    "source_artifact_from_dict",
    "write_json",
    "write_jsonl",
    "write_text",
]

T = TypeVar("T")


def source_artifact_from_dict(raw: Mapping[str, Any]) -> SourceArtifact:
    stored = _mapping(raw.get("object"), "object")
    metadata = _string_mapping(raw.get("metadata", {}), "metadata")
    return SourceArtifact(
        artifact_id=str(raw["artifact_id"]),
        source_id=str(raw["source_id"]),
        source_version=str(raw["source_version"]),
        source_uri=str(raw["source_uri"]),
        object=StoredObject(
            sha256=str(stored["sha256"]),
            uri=str(stored["uri"]),
            byte_size=int(stored["byte_size"]),
        ),
        media_type=str(raw["media_type"]),
        license_id=str(raw["license_id"]),
        access_class=AccessClass(str(raw["access_class"])),
        redistribution=RedistributionPolicy(str(raw["redistribution"])),
        hosted_processing_allowed=bool(raw["hosted_processing_allowed"]),
        retrieved_at=str(raw["retrieved_at"]),
        metadata=metadata,
    )


def document_from_dict(raw: Mapping[str, Any]) -> MinedDocument:
    metadata = _string_mapping(raw.get("metadata", {}), "metadata")
    document = MinedDocument(
        document_id=str(raw["document_id"]),
        text=str(raw["text"]),
        language=str(raw["language"]),
        note_type=str(raw["note_type"]),
        source_artifact_id=str(raw["source_artifact_id"]),
        access_class=AccessClass(str(raw["access_class"])),
        redistribution=RedistributionPolicy(str(raw["redistribution"])),
        hosted_processing_allowed=bool(raw["hosted_processing_allowed"]),
        parent_document_id=(
            None if raw.get("parent_document_id") is None else str(raw["parent_document_id"])
        ),
        group_ids=tuple(str(value) for value in raw.get("group_ids", [])),
        metadata=metadata,
    )
    expected_hash = raw.get("text_sha256")
    if expected_hash is not None and str(expected_hash) != document.text_sha256:
        raise ValueError(f"Text hash mismatch for document {document.document_id!r}")
    return document


def annotation_from_dict(raw: Mapping[str, Any]) -> AnnotationProposal:
    span = _integer_pair(raw.get("span"), "span")
    concepts = raw.get("concepts", [])
    if not isinstance(concepts, list):
        raise ValueError("concepts must be a list")
    parsed_concepts = []
    for value in concepts:
        concept = _mapping(value, "concept")
        parsed_concepts.append(
            ConceptLink(
                code_system=str(concept["code_system"]),
                code=str(concept["code"]),
                terminology_version=str(concept["terminology_version"]),
                confidence=float(concept.get("confidence", 1.0)),
            )
        )
    return AnnotationProposal(
        annotation_id=str(raw["annotation_id"]),
        document_id=str(raw["document_id"]),
        span=span,
        text=str(raw["text"]),
        entity_type=str(raw["entity_type"]),
        assertions=tuple(str(value) for value in raw.get("assertions", [])),
        concepts=tuple(parsed_concepts),
        confidence=float(raw["confidence"]),
        layer=AnnotationLayer(str(raw["layer"])),
        label_source=str(raw["label_source"]),
        labeler_id=str(raw["labeler_id"]),
        review_status=ReviewStatus(str(raw.get("review_status", "proposed"))),
        source_label=(None if raw.get("source_label") is None else str(raw["source_label"])),
        model_revision=(
            None if raw.get("model_revision") is None else str(raw["model_revision"])
        ),
        prompt_hash=None if raw.get("prompt_hash") is None else str(raw["prompt_hash"]),
        metadata=_string_mapping(raw.get("metadata", {}), "metadata"),
    )


def relation_from_dict(raw: Mapping[str, Any]) -> RelationProposal:
    evidence = raw.get("evidence_span")
    return RelationProposal(
        relation_id=str(raw["relation_id"]),
        document_id=str(raw["document_id"]),
        head_annotation_id=str(raw["head_annotation_id"]),
        tail_annotation_id=str(raw["tail_annotation_id"]),
        relation_type=str(raw["relation_type"]),
        confidence=float(raw["confidence"]),
        layer=AnnotationLayer(str(raw["layer"])),
        label_source=str(raw["label_source"]),
        evidence_span=None if evidence is None else _integer_pair(evidence, "evidence_span"),
        labeler_id=None if raw.get("labeler_id") is None else str(raw["labeler_id"]),
        review_status=ReviewStatus(str(raw.get("review_status", "proposed"))),
        metadata=_string_mapping(raw.get("metadata", {}), "metadata"),
    )


def load_source_artifacts(path: str | Path) -> tuple[SourceArtifact, ...]:
    return _load_jsonl(path, source_artifact_from_dict)


def load_documents(path: str | Path) -> tuple[MinedDocument, ...]:
    return _load_jsonl(path, document_from_dict)


def load_annotations(path: str | Path) -> tuple[AnnotationProposal, ...]:
    return _load_jsonl(path, annotation_from_dict)


def load_relations(path: str | Path) -> tuple[RelationProposal, ...]:
    return _load_jsonl(path, relation_from_dict)


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> str:
    """Atomically write sorted-key JSONL and return its SHA-256 fingerprint."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for row in rows:
                encoded = (
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                ).encode("utf-8")
                handle.write(encoded)
                digest.update(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # SCALING: rows never accumulate in memory; replace remains all-or-nothing.
        os.replace(temporary_name, target)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return digest.hexdigest()


def write_json(path: str | Path, payload: Mapping[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write(target, encoded)
    return hashlib.sha256(encoded).hexdigest()


def write_text(path: str | Path, payload: str) -> str:
    """Atomically write UTF-8 text and return its content fingerprint."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = payload.encode("utf-8")
    _atomic_write(target, encoded)
    return hashlib.sha256(encoded).hexdigest()


def _load_jsonl(path: str | Path, parser: Callable[[Mapping[str, Any]], T]) -> tuple[T, ...]:
    values: list[T] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError(f"JSONL row {line_number} must be an object")
            try:
                values.append(parser(raw))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid JSONL row {line_number}: {error}") from error
    return tuple(values)


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _string_mapping(value: Any, field_name: str) -> dict[str, str]:
    mapping = _mapping(value, field_name)
    return {str(key): str(item) for key, item in mapping.items()}


def _integer_pair(value: Any, field_name: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field_name} must be a two-item list")
    return int(value[0]), int(value[1])
