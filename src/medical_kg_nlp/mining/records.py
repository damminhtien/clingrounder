"""Task-neutral, provenance-bearing records for reproducible data mining."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "AccessClass",
    "AnnotationLayer",
    "AnnotationProposal",
    "ConceptLink",
    "CoverageCell",
    "CoverageReport",
    "DatasetSnapshot",
    "DiscoveredArtifact",
    "MinedDocument",
    "RedistributionPolicy",
    "RelationProposal",
    "ReviewStatus",
    "SourceArtifact",
    "SourceRequest",
    "StoredObject",
]


class AccessClass(str, Enum):
    """Operational access level used by privacy and hosted-processing gates."""

    OPEN = "open"
    OPEN_WITH_TERMS = "open_with_terms"
    CREDENTIALLED = "credentialled"
    DUA = "dua"
    LOCAL_PRIVATE = "local_private"
    QUARANTINE = "quarantine"


class RedistributionPolicy(str, Enum):
    """How source-derived records may leave the controlled data plane."""

    ALLOWED = "allowed"
    ATTRIBUTION = "attribution"
    NON_COMMERCIAL = "non_commercial"
    PROHIBITED = "prohibited"
    UNKNOWN = "unknown"


class AnnotationLayer(str, Enum):
    """Quality and lifecycle layer for an annotation set."""

    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    CHALLENGE = "challenge"


class ReviewStatus(str, Enum):
    """Human-review state for a model or rule proposal."""

    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


@dataclass(frozen=True)
class StoredObject:
    """One immutable object addressed by its SHA-256 digest."""

    sha256: str
    uri: str
    byte_size: int

    def __post_init__(self) -> None:
        _validate_sha256(self.sha256, field_name="stored object sha256")
        if self.byte_size < 0:
            raise ValueError("Stored object byte_size must be non-negative")
        if not self.uri.strip():
            raise ValueError("Stored object uri must be non-empty")


@dataclass(frozen=True)
class SourceRequest:
    """A version-pinned discovery request for one registered source."""

    source_id: str
    source_version: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.source_id, "source_id")
        _require_text(self.source_version, "source_version")


@dataclass(frozen=True)
class DiscoveredArtifact:
    """Remote or local artifact metadata known before bytes are materialized."""

    source_id: str
    source_version: str
    uri: str
    media_type: str
    expected_sha256: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("source_id", self.source_id),
            ("source_version", self.source_version),
            ("uri", self.uri),
            ("media_type", self.media_type),
        ):
            _require_text(value, name)
        if self.expected_sha256 is not None:
            _validate_sha256(self.expected_sha256, field_name="expected_sha256")


@dataclass(frozen=True)
class SourceArtifact:
    """Materialized source bytes plus the policy needed for downstream use."""

    artifact_id: str
    source_id: str
    source_version: str
    source_uri: str
    object: StoredObject
    media_type: str
    license_id: str
    access_class: AccessClass
    redistribution: RedistributionPolicy
    hosted_processing_allowed: bool
    retrieved_at: str
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("artifact_id", self.artifact_id),
            ("source_id", self.source_id),
            ("source_version", self.source_version),
            ("source_uri", self.source_uri),
            ("media_type", self.media_type),
            ("license_id", self.license_id),
            ("retrieved_at", self.retrieved_at),
        ):
            _require_text(value, name)
        # PRIVACY: restricted material must never be eligible for hosted processing.
        if self.access_class in {
            AccessClass.CREDENTIALLED,
            AccessClass.DUA,
            AccessClass.LOCAL_PRIVATE,
            AccessClass.QUARANTINE,
        } and self.hosted_processing_allowed:
            raise ValueError(
                f"{self.access_class.value} artifacts cannot allow hosted processing"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "source_uri": self.source_uri,
            "object": {
                "sha256": self.object.sha256,
                "uri": self.object.uri,
                "byte_size": self.object.byte_size,
            },
            "media_type": self.media_type,
            "license_id": self.license_id,
            "access_class": self.access_class.value,
            "redistribution": self.redistribution.value,
            "hosted_processing_allowed": self.hosted_processing_allowed,
            "retrieved_at": self.retrieved_at,
            "metadata": dict(sorted(self.metadata.items())),
        }


@dataclass(frozen=True)
class MinedDocument:
    """An immutable text unit whose annotations always target ``text`` offsets."""

    document_id: str
    text: str
    language: str
    note_type: str
    source_artifact_id: str
    access_class: AccessClass
    redistribution: RedistributionPolicy
    hosted_processing_allowed: bool
    parent_document_id: str | None = None
    group_ids: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("document_id", self.document_id),
            ("text", self.text),
            ("language", self.language),
            ("note_type", self.note_type),
            ("source_artifact_id", self.source_artifact_id),
        ):
            _require_text(value, name)
        if self.parent_document_id == self.document_id:
            raise ValueError("A mined document cannot be its own parent")
        if any(not value.strip() for value in self.group_ids):
            raise ValueError("group_ids must contain non-empty values")
        if self.access_class in {
            AccessClass.CREDENTIALLED,
            AccessClass.DUA,
            AccessClass.LOCAL_PRIVATE,
            AccessClass.QUARANTINE,
        } and self.hosted_processing_allowed:
            raise ValueError("Restricted documents cannot allow hosted processing")

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "language": self.language,
            "note_type": self.note_type,
            "source_artifact_id": self.source_artifact_id,
            "access_class": self.access_class.value,
            "redistribution": self.redistribution.value,
            "hosted_processing_allowed": self.hosted_processing_allowed,
            "parent_document_id": self.parent_document_id,
            "group_ids": list(self.group_ids),
            "metadata": dict(sorted(self.metadata.items())),
        }


@dataclass(frozen=True)
class ConceptLink:
    """A terminology-versioned concept identifier attached to a proposal."""

    code_system: str
    code: str
    terminology_version: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("code_system", self.code_system),
            ("code", self.code),
            ("terminology_version", self.terminology_version),
        ):
            _require_text(value, name)
        _validate_probability(self.confidence, "concept confidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_system": self.code_system,
            "code": self.code,
            "terminology_version": self.terminology_version,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class AnnotationProposal:
    """One provenance-bearing entity annotation in a named quality layer."""

    annotation_id: str
    document_id: str
    span: tuple[int, int]
    text: str
    entity_type: str
    assertions: tuple[str, ...]
    concepts: tuple[ConceptLink, ...]
    confidence: float
    layer: AnnotationLayer
    label_source: str
    labeler_id: str
    review_status: ReviewStatus = ReviewStatus.PROPOSED
    source_label: str | None = None
    model_revision: str | None = None
    prompt_hash: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("annotation_id", self.annotation_id),
            ("document_id", self.document_id),
            ("text", self.text),
            ("entity_type", self.entity_type),
            ("label_source", self.label_source),
            ("labeler_id", self.labeler_id),
        ):
            _require_text(value, name)
        start, end = self.span
        if start < 0 or end <= start:
            raise ValueError(f"Invalid annotation span {self.span}")
        _validate_probability(self.confidence, "annotation confidence")
        if self.prompt_hash is not None:
            _validate_sha256(self.prompt_hash, field_name="prompt_hash")

    def validate_offsets(self, document: MinedDocument) -> None:
        """Validate the non-negotiable raw-text offset invariant."""

        if self.document_id != document.document_id:
            raise ValueError("Annotation and document IDs do not match")
        start, end = self.span
        if end > len(document.text) or document.text[start:end] != self.text:
            actual = document.text[start:end] if end <= len(document.text) else "<out-of-range>"
            raise ValueError(f"Offset mismatch for {self.annotation_id}: {actual!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "document_id": self.document_id,
            "span": [self.span[0], self.span[1]],
            "text": self.text,
            "entity_type": self.entity_type,
            "assertions": list(self.assertions),
            "concepts": [item.to_dict() for item in self.concepts],
            "confidence": self.confidence,
            "layer": self.layer.value,
            "label_source": self.label_source,
            "labeler_id": self.labeler_id,
            "review_status": self.review_status.value,
            "source_label": self.source_label,
            "model_revision": self.model_revision,
            "prompt_hash": self.prompt_hash,
            "metadata": dict(sorted(self.metadata.items())),
        }


@dataclass(frozen=True)
class RelationProposal:
    """A typed relation between proposals from the same mined document."""

    relation_id: str
    document_id: str
    head_annotation_id: str
    tail_annotation_id: str
    relation_type: str
    confidence: float
    layer: AnnotationLayer
    label_source: str
    evidence_span: tuple[int, int] | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("relation_id", self.relation_id),
            ("document_id", self.document_id),
            ("head_annotation_id", self.head_annotation_id),
            ("tail_annotation_id", self.tail_annotation_id),
            ("relation_type", self.relation_type),
            ("label_source", self.label_source),
        ):
            _require_text(value, name)
        if self.head_annotation_id == self.tail_annotation_id:
            raise ValueError("Relation endpoints must be different")
        _validate_probability(self.confidence, "relation confidence")
        if self.evidence_span is not None:
            start, end = self.evidence_span
            if start < 0 or end <= start:
                raise ValueError(f"Invalid relation evidence span {self.evidence_span}")

    def validate(self, document: MinedDocument, annotations: dict[str, AnnotationProposal]) -> None:
        if self.document_id != document.document_id:
            raise ValueError("Relation and document IDs do not match")
        for annotation_id in (self.head_annotation_id, self.tail_annotation_id):
            annotation = annotations.get(annotation_id)
            if annotation is None or annotation.document_id != self.document_id:
                raise ValueError(f"Invalid relation endpoint {annotation_id!r}")
        if self.evidence_span is not None and self.evidence_span[1] > len(document.text):
            raise ValueError("Relation evidence span exceeds source text")

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "document_id": self.document_id,
            "head_annotation_id": self.head_annotation_id,
            "tail_annotation_id": self.tail_annotation_id,
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "layer": self.layer.value,
            "label_source": self.label_source,
            "evidence_span": list(self.evidence_span) if self.evidence_span else None,
            "metadata": dict(sorted(self.metadata.items())),
        }


@dataclass(frozen=True)
class CoverageCell:
    """Observed and target support for one multidimensional coverage slice."""

    dimensions: tuple[tuple[str, str], ...]
    observed: int
    target: int
    human_reviewed: int = 0
    synthetic: int = 0

    def __post_init__(self) -> None:
        if self.observed < 0 or self.target < 0 or self.human_reviewed < 0 or self.synthetic < 0:
            raise ValueError("Coverage counts must be non-negative")
        if any(not key.strip() or not value.strip() for key, value in self.dimensions):
            raise ValueError("Coverage dimensions must contain non-empty keys and values")

    @property
    def gap_ratio(self) -> float:
        if self.target == 0:
            return 0.0
        return max(0.0, 1.0 - min(self.observed / self.target, 1.0))


@dataclass(frozen=True)
class CoverageReport:
    """Coverage cells and aggregate deficit for a frozen snapshot."""

    snapshot_id: str
    cells: tuple[CoverageCell, ...]

    @property
    def mean_gap(self) -> float:
        if not self.cells:
            return 0.0
        return sum(cell.gap_ratio for cell in self.cells) / len(self.cells)


@dataclass(frozen=True)
class DatasetSnapshot:
    """Immutable manifest summary for a reproducible dataset release."""

    snapshot_id: str
    version: str
    manifest_sha256: str
    document_count: int
    annotation_count: int
    relation_count: int
    source_fingerprints: tuple[str, ...]
    split_counts: tuple[tuple[str, int], ...]
    redistributable: bool
    created_at: str
    restricted_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("snapshot_id", self.snapshot_id),
            ("version", self.version),
            ("created_at", self.created_at),
        ):
            _require_text(value, name)
        _validate_sha256(self.manifest_sha256, field_name="manifest_sha256")
        if min(self.document_count, self.annotation_count, self.relation_count) < 0:
            raise ValueError("Snapshot counts must be non-negative")
        if self.redistributable and self.restricted_reasons:
            raise ValueError("Redistributable snapshots cannot have restricted reasons")

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "version": self.version,
            "manifest_sha256": self.manifest_sha256,
            "document_count": self.document_count,
            "annotation_count": self.annotation_count,
            "relation_count": self.relation_count,
            "source_fingerprints": list(self.source_fingerprints),
            "split_counts": dict(self.split_counts),
            "redistributable": self.redistributable,
            "created_at": self.created_at,
            "restricted_reasons": list(self.restricted_reasons),
        }


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _validate_probability(value: float, field_name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be finite and between 0 and 1")


def _validate_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
