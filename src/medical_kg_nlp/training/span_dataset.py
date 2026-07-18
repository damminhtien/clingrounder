"""Validated, streaming access to model-neutral mined span datasets.

The mining package owns dataset construction. This module is the training-side
contract: it rejects changed text, invalid local/source offsets, duplicate labels,
and split drift before an optional ML framework is imported.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.utils.hashing import sha256_file, sha256_text

__all__ = [
    "SpanDatasetSummary",
    "SpanTrainingEntity",
    "SpanTrainingRecord",
    "build_bio_label_vocabulary",
    "iter_span_training_records",
    "scan_span_dataset",
    "validate_span_dataset_manifest",
]

_SPAN_DATASET_SCHEMA = "mined-span-dataset.v1"


@dataclass(frozen=True)
class SpanTrainingEntity:
    """One entity expressed in both chunk-local and immutable source coordinates."""

    annotation_id: str
    start: int
    end: int
    source_start: int
    source_end: int
    text: str
    label: str


@dataclass(frozen=True)
class SpanTrainingRecord:
    """One raw-text training chunk with non-overlapping entity annotations."""

    record_id: str
    document_id: str
    split: str
    text: str
    text_sha256: str
    source_span: tuple[int, int]
    language: str
    note_type: str
    source_artifact_id: str
    entities: tuple[SpanTrainingEntity, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SpanTrainingRecord":
        """Parse and validate one JSON-compatible mined span record."""

        raw_entities = raw.get("entities")
        if not isinstance(raw_entities, list):
            raise ValueError("entities must be a list")
        entities = tuple(_entity_from_mapping(value) for value in raw_entities)
        record = cls(
            record_id=_required_string(raw, "record_id"),
            document_id=_required_string(raw, "document_id"),
            split=_required_string(raw, "split"),
            text=_required_string(raw, "text", allow_empty=True),
            text_sha256=_required_string(raw, "text_sha256"),
            source_span=_integer_pair(raw.get("source_span"), "source_span"),
            language=_required_string(raw, "language"),
            note_type=_required_string(raw, "note_type"),
            source_artifact_id=_required_string(raw, "source_artifact_id"),
            entities=entities,
        )
        record.validate()
        return record

    def validate(self) -> None:
        """Enforce lossless source/local offset and BIO compatibility invariants."""

        if sha256_text(self.text) != self.text_sha256:
            raise ValueError(f"Text hash mismatch for record {self.record_id!r}")
        source_start, source_end = self.source_span
        if source_start < 0 or source_end < source_start:
            raise ValueError(f"Invalid source_span for record {self.record_id!r}")
        if source_end - source_start != len(self.text):
            raise ValueError(
                f"source_span length does not match text for record {self.record_id!r}"
            )

        ordered = sorted(
            self.entities,
            key=lambda item: (item.start, item.end, item.annotation_id),
        )
        for entity in ordered:
            if not 0 <= entity.start < entity.end <= len(self.text):
                raise ValueError(f"Invalid span for annotation {entity.annotation_id!r}")
            # INVARIANT: every training label must remain an exact raw chunk slice.
            if self.text[entity.start : entity.end] != entity.text:
                raise ValueError(f"Offset mismatch for annotation {entity.annotation_id!r}")
            if source_start + entity.start != entity.source_start:
                raise ValueError(f"Source start mismatch for annotation {entity.annotation_id!r}")
            if source_start + entity.end != entity.source_end:
                raise ValueError(f"Source end mismatch for annotation {entity.annotation_id!r}")
        for left, right in zip(ordered, ordered[1:], strict=False):
            if right.start < left.end:
                raise ValueError(
                    "Overlapping entities are not BIO-compatible: "
                    f"{left.annotation_id!r} and {right.annotation_id!r}"
                )


@dataclass(frozen=True)
class SpanDatasetSummary:
    """Dataset identity and label distribution collected without loading all rows."""

    dataset_sha256: str
    record_count: int
    entity_count: int
    empty_record_count: int
    split_record_counts: Mapping[str, int]
    split_entity_counts: Mapping[str, int]
    labels_by_split: Mapping[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation for run manifests."""

        return {
            "dataset_sha256": self.dataset_sha256,
            "record_count": self.record_count,
            "entity_count": self.entity_count,
            "empty_record_count": self.empty_record_count,
            "split_record_counts": dict(sorted(self.split_record_counts.items())),
            "split_entity_counts": dict(sorted(self.split_entity_counts.items())),
            "labels_by_split": {
                split: list(labels)
                for split, labels in sorted(self.labels_by_split.items())
            },
        }


def iter_span_training_records(path: str | Path) -> Iterator[SpanTrainingRecord]:
    """Stream strict training records from JSONL with useful row diagnostics."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, Mapping):
                    raise ValueError("row must be an object")
                yield SpanTrainingRecord.from_mapping(raw)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error


def scan_span_dataset(path: str | Path) -> SpanDatasetSummary:
    """Validate a complete dataset and summarize its immutable split distribution."""

    source = Path(path)
    record_ids: set[str] = set()
    annotation_ids: set[str] = set()
    split_records: Counter[str] = Counter()
    split_entities: Counter[str] = Counter()
    labels: dict[str, set[str]] = defaultdict(set)
    entity_count = 0
    empty_record_count = 0

    for record in iter_span_training_records(source):
        if record.record_id in record_ids:
            raise ValueError(f"Duplicate record ID {record.record_id!r}")
        record_ids.add(record.record_id)
        split_records[record.split] += 1
        if not record.entities:
            empty_record_count += 1
        for entity in record.entities:
            if entity.annotation_id in annotation_ids:
                raise ValueError(f"Duplicate annotation ID {entity.annotation_id!r}")
            annotation_ids.add(entity.annotation_id)
            split_entities[record.split] += 1
            labels[record.split].add(entity.label)
            entity_count += 1

    if not record_ids:
        raise ValueError("Span dataset must contain at least one record")
    return SpanDatasetSummary(
        dataset_sha256=sha256_file(source),
        record_count=len(record_ids),
        entity_count=entity_count,
        empty_record_count=empty_record_count,
        split_record_counts=dict(sorted(split_records.items())),
        split_entity_counts=dict(sorted(split_entities.items())),
        labels_by_split={
            split: tuple(sorted(values)) for split, values in sorted(labels.items())
        },
    )


def validate_span_dataset_manifest(
    dataset_path: str | Path,
    manifest_path: str | Path,
    summary: SpanDatasetSummary | None = None,
) -> Mapping[str, Any]:
    """Verify that a mined dataset still matches its pinned construction manifest."""

    source = Path(manifest_path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{source}: manifest must be an object")
    if raw.get("schema_version") != _SPAN_DATASET_SCHEMA:
        raise ValueError(f"{source}: unsupported span dataset schema")
    expected_sha256 = str(raw.get("output_sha256", ""))
    observed = summary or scan_span_dataset(dataset_path)
    if not expected_sha256 or expected_sha256 != observed.dataset_sha256:
        raise ValueError(
            "Span dataset fingerprint differs from its manifest: "
            f"expected {expected_sha256 or '<missing>'}, observed {observed.dataset_sha256}"
        )
    expected_records = raw.get("chunk_count")
    if isinstance(expected_records, int) and expected_records != observed.record_count:
        raise ValueError("Span dataset record count differs from its manifest")
    expected_entities = raw.get("entity_count")
    if isinstance(expected_entities, int) and expected_entities != observed.entity_count:
        raise ValueError("Span dataset entity count differs from its manifest")
    return raw


def build_bio_label_vocabulary(
    summary: SpanDatasetSummary,
    *,
    train_split: str,
    evaluation_split: str | None = None,
) -> tuple[str, ...]:
    """Create a deterministic train-only BIO vocabulary and reject unseen eval labels."""

    train_labels = summary.labels_by_split.get(train_split, ())
    if not train_labels:
        raise ValueError(f"Training split {train_split!r} contains no entity labels")
    if evaluation_split is not None:
        evaluation_labels = set(summary.labels_by_split.get(evaluation_split, ()))
        unseen = sorted(evaluation_labels - set(train_labels))
        if unseen:
            raise ValueError(
                f"Evaluation split {evaluation_split!r} has unseen labels: {unseen}"
            )
    output = ["O"]
    for label in sorted(train_labels):
        output.extend((f"B-{label}", f"I-{label}"))
    return tuple(output)


def _entity_from_mapping(raw: object) -> SpanTrainingEntity:
    if not isinstance(raw, Mapping):
        raise ValueError("entity must be an object")
    return SpanTrainingEntity(
        annotation_id=_required_string(raw, "annotation_id"),
        start=_required_int(raw, "start"),
        end=_required_int(raw, "end"),
        source_start=_required_int(raw, "source_start"),
        source_end=_required_int(raw, "source_end"),
        text=_required_string(raw, "text"),
        label=_required_string(raw, "label"),
    )


def _required_string(
    raw: Mapping[str, Any],
    key: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_int(raw: Mapping[str, Any], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _integer_pair(value: object, field_name: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field_name} must be a two-item list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{field_name} values must be integers")
    return int(value[0]), int(value[1])
