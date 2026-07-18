"""Provenance-bearing mention inventories for terminology and ontology review."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from medical_kg_nlp.mining.policy import MiningQualityGate
from medical_kg_nlp.mining.records import AnnotationProposal, MinedDocument
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = [
    "MentionInventoryEntry",
    "MentionInventoryResult",
    "build_mention_inventory",
]


@dataclass(frozen=True)
class MentionInventoryEntry:
    """One normalized mention/type hypothesis backed by source occurrences."""

    term_id: str
    normalized_mention: str
    entity_type: str
    source_label: str | None
    occurrence_count: int
    document_count: int
    consensus_occurrence_count: int
    surface_variant_count: int
    surface_variants: tuple[tuple[str, int], ...]
    source_artifact_ids: tuple[str, ...]
    label_sources: tuple[tuple[str, int], ...]
    example_document_ids: tuple[str, ...]
    review_tier: str
    recommended_use: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "term_id": self.term_id,
            "normalized_mention": self.normalized_mention,
            "entity_type": self.entity_type,
            "source_label": self.source_label,
            "occurrence_count": self.occurrence_count,
            "document_count": self.document_count,
            "consensus_occurrence_count": self.consensus_occurrence_count,
            "surface_variant_count": self.surface_variant_count,
            "surface_variants": [
                {"surface": surface, "count": count} for surface, count in self.surface_variants
            ],
            "source_artifact_ids": list(self.source_artifact_ids),
            "label_sources": dict(self.label_sources),
            "example_document_ids": list(self.example_document_ids),
            "review_tier": self.review_tier,
            "recommended_use": self.recommended_use,
            # INVARIANT: mined mentions are not terminology concepts until a pinned
            # repository provides a unique, type-compatible reviewed mapping.
            "concept_status": "unlinked",
            "concepts": [],
        }


@dataclass(frozen=True)
class MentionInventoryResult:
    """Inventory rows, ambiguity conflicts, and aggregate audit metrics."""

    entries: tuple[MentionInventoryEntry, ...]
    conflicts: tuple[dict[str, Any], ...]
    report: dict[str, Any]


def build_mention_inventory(
    documents: Sequence[MinedDocument],
    annotations: Sequence[AnnotationProposal],
    *,
    min_occurrences: int = 1,
    min_documents: int = 1,
    max_surface_variants: int = 20,
    max_examples: int = 5,
) -> MentionInventoryResult:
    """Aggregate annotation surfaces without inventing canonical terms or codes."""

    if min_occurrences <= 0 or min_documents <= 0:
        raise ValueError("Mention inventory thresholds must be positive")
    if max_surface_variants <= 0 or max_examples <= 0:
        raise ValueError("Mention inventory output limits must be positive")
    issues = MiningQualityGate().validate(documents, annotations)
    if issues:
        raise ValueError("Cannot inventory invalid mined data:\n" + "\n".join(issues))

    documents_by_id = {document.document_id: document for document in documents}
    grouped: dict[tuple[str, str, str], list[AnnotationProposal]] = defaultdict(list)
    skipped_empty_mentions = 0
    for annotation in annotations:
        normalized = normalize_for_match(annotation.text)
        if not normalized:
            skipped_empty_mentions += 1
            continue
        grouped[(normalized, annotation.entity_type, annotation.source_label or "")].append(
            annotation
        )

    entries: list[MentionInventoryEntry] = []
    filtered_group_count = 0
    for (normalized, entity_type, source_label), mention_annotations in sorted(grouped.items()):
        document_ids = sorted({annotation.document_id for annotation in mention_annotations})
        if len(mention_annotations) < min_occurrences or len(document_ids) < min_documents:
            filtered_group_count += 1
            continue
        surfaces = Counter(annotation.text for annotation in mention_annotations)
        label_sources = Counter(annotation.label_source for annotation in mention_annotations)
        source_artifacts = sorted(
            {documents_by_id[document_id].source_artifact_id for document_id in document_ids}
        )
        term_id = _term_id(normalized, entity_type, source_label)
        consensus_count = sum(
            annotation.metadata.get("consensus") == "true"
            or annotation.label_source == "exact_duplicate_consensus"
            for annotation in mention_annotations
        )
        entries.append(
            MentionInventoryEntry(
                term_id=term_id,
                normalized_mention=normalized,
                entity_type=entity_type,
                source_label=source_label or None,
                occurrence_count=len(mention_annotations),
                document_count=len(document_ids),
                consensus_occurrence_count=consensus_count,
                surface_variant_count=len(surfaces),
                surface_variants=tuple(
                    sorted(surfaces.items(), key=lambda item: (-item[1], item[0]))[
                        :max_surface_variants
                    ]
                ),
                source_artifact_ids=tuple(source_artifacts),
                label_sources=tuple(sorted(label_sources.items())),
                example_document_ids=tuple(document_ids[:max_examples]),
                review_tier=_review_tier(
                    occurrence_count=len(mention_annotations),
                    document_count=len(document_ids),
                    consensus_count=consensus_count,
                ),
                recommended_use=(
                    "context_metadata_review"
                    if entity_type == "OTHER"
                    else "terminology_alias_review"
                ),
            )
        )

    entries_by_mention: dict[str, list[MentionInventoryEntry]] = defaultdict(list)
    for entry in entries:
        entries_by_mention[entry.normalized_mention].append(entry)
    conflicts = []
    for normalized, mention_entries in sorted(entries_by_mention.items()):
        semantic_keys = {(entry.entity_type, entry.source_label) for entry in mention_entries}
        if len(semantic_keys) <= 1:
            continue
        conflicts.append(
            {
                "conflict_id": "mention-conflict:"
                + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24],
                "normalized_mention": normalized,
                "conflict_types": _conflict_types(mention_entries),
                "hypotheses": [
                    {
                        "term_id": entry.term_id,
                        "entity_type": entry.entity_type,
                        "source_label": entry.source_label,
                        "occurrence_count": entry.occurrence_count,
                        "document_count": entry.document_count,
                    }
                    for entry in sorted(
                        mention_entries,
                        key=lambda item: (
                            item.entity_type,
                            item.source_label or "",
                            item.term_id,
                        ),
                    )
                ],
                "resolution": "human_or_terminology_review_required",
            }
        )

    ordered_entries = tuple(
        sorted(
            entries,
            key=lambda item: (
                -item.document_count,
                -item.occurrence_count,
                item.normalized_mention,
                item.entity_type,
                item.source_label or "",
            ),
        )
    )
    report = {
        "schema_version": "medical-mention-inventory.v1",
        "document_count": len(documents),
        "annotation_count": len(annotations),
        "entry_count": len(ordered_entries),
        "ambiguous_mention_count": len(conflicts),
        "skipped_empty_mention_count": skipped_empty_mentions,
        "filtered_group_count": filtered_group_count,
        "thresholds": {
            "min_occurrences": min_occurrences,
            "min_documents": min_documents,
        },
        "entity_types": dict(
            sorted(Counter(entry.entity_type for entry in ordered_entries).items())
        ),
        "source_labels": dict(
            sorted(Counter(entry.source_label or "<none>" for entry in ordered_entries).items())
        ),
        "review_tiers": dict(
            sorted(Counter(entry.review_tier for entry in ordered_entries).items())
        ),
    }
    return MentionInventoryResult(
        entries=ordered_entries,
        conflicts=tuple(conflicts),
        report=report,
    )


def _term_id(normalized: str, entity_type: str, source_label: str) -> str:
    identity = f"{normalized}\0{entity_type}\0{source_label}".encode("utf-8")
    return f"mined-term:{hashlib.sha256(identity).hexdigest()[:24]}"


def _review_tier(*, occurrence_count: int, document_count: int, consensus_count: int) -> str:
    if consensus_count:
        return "duplicate_consensus_supported"
    if document_count >= 2:
        return "multi_document"
    if occurrence_count >= 2:
        return "repeated_single_document"
    return "singleton"


def _conflict_types(entries: Sequence[MentionInventoryEntry]) -> list[str]:
    result = []
    if len({entry.entity_type for entry in entries}) > 1:
        result.append("entity_type_conflict")
    if len({entry.source_label for entry in entries}) > 1:
        result.append("source_label_conflict")
    return result
