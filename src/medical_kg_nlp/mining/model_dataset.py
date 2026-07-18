"""Model-neutral span datasets compiled from mined documents and annotations."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.mining.io import write_json, write_jsonl
from medical_kg_nlp.mining.records import AnnotationProposal, MinedDocument
from medical_kg_nlp.utils.hashing import sha256_file, sha256_text

__all__ = [
    "SpanDatasetConfig",
    "export_span_dataset",
    "iter_span_training_records",
    "load_dataset_splits",
]


@dataclass(frozen=True)
class SpanDatasetConfig:
    """Controls lossless character-window generation for token classification."""

    max_characters: int = 1600
    entity_types: tuple[str, ...] = ()
    include_empty_chunks: bool = True
    empty_chunk_rate: float = 1.0

    def __post_init__(self) -> None:
        if self.max_characters < 128:
            raise ValueError("max_characters must be at least 128")
        if any(not value.strip() for value in self.entity_types):
            raise ValueError("entity_types must contain non-empty values")
        if not 0.0 <= self.empty_chunk_rate <= 1.0:
            raise ValueError("empty_chunk_rate must be in [0, 1]")


def load_dataset_splits(path: str | Path) -> dict[str, str]:
    """Load the immutable document-to-split mapping from a snapshot manifest."""

    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("splits"), Mapping):
        raise ValueError(f"{source}: snapshot manifest must contain a splits mapping")
    splits = {str(key): str(value) for key, value in raw["splits"].items()}
    if not splits or any(not key.strip() or not value.strip() for key, value in splits.items()):
        raise ValueError(f"{source}: splits must contain non-empty document IDs and names")
    return splits


def iter_span_training_records(
    documents: Sequence[MinedDocument],
    annotations: Sequence[AnnotationProposal],
    splits: Mapping[str, str],
    config: SpanDatasetConfig,
) -> Iterator[dict[str, Any]]:
    """Yield deterministic raw-text chunks with local and source entity offsets.

    BIO token classification cannot represent overlapping labels. This function
    therefore rejects overlaps before writing any artifact, while preserving every
    accepted annotation exactly once in a non-overlapping character window.
    """

    documents_by_id = _unique_documents(documents)
    grouped = _validated_annotations(documents_by_id, annotations, config)
    missing_splits = sorted(set(documents_by_id) - set(splits))
    if missing_splits:
        raise ValueError(f"Missing split assignments for {len(missing_splits)} documents")

    for document in sorted(documents, key=lambda item: (splits[item.document_id], item.document_id)):
        document_annotations = grouped.get(document.document_id, ())
        for start, end in _chunk_spans(
            document.text,
            document_annotations,
            max_characters=config.max_characters,
        ):
            chunk_annotations = [
                annotation
                for annotation in document_annotations
                if start <= annotation.span[0] and annotation.span[1] <= end
            ]
            identity = f"{document.document_id}\x1f{start}\x1f{end}"
            if not chunk_annotations and (
                not config.include_empty_chunks
                or not _sample_empty_chunk(identity, config.empty_chunk_rate)
            ):
                continue
            text = document.text[start:end]
            record = {
                "record_id": f"span-record:{sha256_text(identity)[:24]}",
                "document_id": document.document_id,
                "split": splits[document.document_id],
                "text": text,
                "text_sha256": sha256_text(text),
                "source_span": [start, end],
                "language": document.language,
                "note_type": document.note_type,
                "source_artifact_id": document.source_artifact_id,
                "entities": [
                    {
                        "annotation_id": annotation.annotation_id,
                        "start": annotation.span[0] - start,
                        "end": annotation.span[1] - start,
                        "source_start": annotation.span[0],
                        "source_end": annotation.span[1],
                        "text": annotation.text,
                        "label": annotation.entity_type,
                    }
                    for annotation in chunk_annotations
                ],
            }
            _validate_training_record(record, document)
            yield record


def export_span_dataset(
    documents: Sequence[MinedDocument],
    annotations: Sequence[AnnotationProposal],
    splits: Mapping[str, str],
    config: SpanDatasetConfig,
    *,
    output_path: str | Path,
    manifest_path: str | Path,
    documents_path: str | Path,
    annotations_path: str | Path,
    split_manifest_path: str | Path,
) -> dict[str, Any]:
    """Stream a deterministic JSONL dataset and write its pinned manifest."""

    split_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    chunk_count = 0
    empty_chunk_count = 0
    entity_count = 0

    def observed_records() -> Iterator[dict[str, Any]]:
        nonlocal chunk_count, empty_chunk_count, entity_count
        for record in iter_span_training_records(documents, annotations, splits, config):
            chunk_count += 1
            split_counts[str(record["split"])] += 1
            entities = record["entities"]
            if not isinstance(entities, list):
                raise TypeError("Training record entities must be a list")
            if not entities:
                empty_chunk_count += 1
            entity_count += len(entities)
            for entity in entities:
                if not isinstance(entity, Mapping):
                    raise TypeError("Training record entity must be an object")
                type_counts[str(entity["label"])] += 1
            yield record

    output_sha256 = write_jsonl(output_path, observed_records())
    manifest: dict[str, Any] = {
        "schema_version": "mined-span-dataset.v1",
        "document_count": len(documents),
        "chunk_count": chunk_count,
        "empty_chunk_count": empty_chunk_count,
        "entity_count": entity_count,
        "split_chunk_counts": dict(sorted(split_counts.items())),
        "entity_type_counts": dict(sorted(type_counts.items())),
        "config": {
            "max_characters": config.max_characters,
            "entity_types": list(config.entity_types),
            "include_empty_chunks": config.include_empty_chunks,
            "empty_chunk_rate": config.empty_chunk_rate,
        },
        "inputs": {
            "documents": _fingerprinted_path(documents_path),
            "annotations": _fingerprinted_path(annotations_path),
            "split_manifest": _fingerprinted_path(split_manifest_path),
        },
        "output": str(Path(output_path)),
        "output_sha256": output_sha256,
    }
    write_json(manifest_path, manifest)
    return manifest


def _unique_documents(documents: Sequence[MinedDocument]) -> dict[str, MinedDocument]:
    output: dict[str, MinedDocument] = {}
    for document in documents:
        if document.document_id in output:
            raise ValueError(f"Duplicate document ID {document.document_id!r}")
        output[document.document_id] = document
    if not output:
        raise ValueError("At least one document is required")
    return output


def _validated_annotations(
    documents: Mapping[str, MinedDocument],
    annotations: Sequence[AnnotationProposal],
    config: SpanDatasetConfig,
) -> dict[str, tuple[AnnotationProposal, ...]]:
    accepted_types = set(config.entity_types)
    grouped: dict[str, list[AnnotationProposal]] = defaultdict(list)
    annotation_ids: set[str] = set()
    for annotation in annotations:
        if accepted_types and annotation.entity_type not in accepted_types:
            continue
        if annotation.annotation_id in annotation_ids:
            raise ValueError(f"Duplicate annotation ID {annotation.annotation_id!r}")
        annotation_ids.add(annotation.annotation_id)
        document = documents.get(annotation.document_id)
        if document is None:
            raise ValueError(f"Unknown annotation document {annotation.document_id!r}")
        annotation.validate_offsets(document)
        grouped[annotation.document_id].append(annotation)

    output: dict[str, tuple[AnnotationProposal, ...]] = {}
    for document_id, values in grouped.items():
        ordered = sorted(values, key=lambda item: (item.span[0], item.span[1], item.annotation_id))
        for left, right in zip(ordered, ordered[1:], strict=False):
            if right.span[0] < left.span[1]:
                raise ValueError(
                    "Overlapping annotations are not BIO-compatible: "
                    f"{left.annotation_id!r} and {right.annotation_id!r}"
                )
        output[document_id] = tuple(ordered)
    return output


def _chunk_spans(
    text: str,
    annotations: Sequence[AnnotationProposal],
    *,
    max_characters: int,
) -> Iterator[tuple[int, int]]:
    start = 0
    while start < len(text):
        end = min(len(text), start + max_characters)
        if end < len(text):
            preferred = _preferred_boundary(text, start, end)
            if preferred > start:
                end = preferred
            # INVARIANT: a model chunk may grow past its soft limit, but never cuts a label.
            while True:
                crossing = next(
                    (
                        annotation
                        for annotation in annotations
                        if annotation.span[0] < end < annotation.span[1]
                    ),
                    None,
                )
                if crossing is None:
                    break
                end = crossing.span[1]
        if end <= start:
            raise RuntimeError("Span dataset chunking did not make progress")
        yield start, end
        start = end


def _sample_empty_chunk(identity: str, rate: float) -> bool:
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    bucket = int(sha256_text(identity)[:8], 16) / 0xFFFFFFFF
    # SCALING: hash sampling is deterministic and does not buffer negative chunks.
    return bucket < rate


def _preferred_boundary(text: str, start: int, end: int) -> int:
    minimum = start + int((end - start) * 0.7)
    for separator in ("\n\n", "\n", ". ", "; ", " "):
        index = text.rfind(separator, minimum, end)
        if index >= minimum:
            return index + len(separator)
    return end


def _validate_training_record(record: Mapping[str, Any], document: MinedDocument) -> None:
    source_start, source_end = (int(value) for value in record["source_span"])
    text = str(record["text"])
    if document.text[source_start:source_end] != text:
        raise ValueError("Training chunk does not match its source document")
    for raw in record["entities"]:
        start = int(raw["start"])
        end = int(raw["end"])
        if text[start:end] != str(raw["text"]):
            raise ValueError(f"Training entity offset mismatch for {raw['annotation_id']!r}")
        if source_start + start != int(raw["source_start"]):
            raise ValueError("Training entity start does not round-trip to the source document")
        if source_start + end != int(raw["source_end"]):
            raise ValueError("Training entity end does not round-trip to the source document")


def _fingerprinted_path(path: str | Path) -> dict[str, str]:
    return {"path": str(Path(path)), "sha256": sha256_file(path)}
