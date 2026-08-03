"""Adapt reviewed Phase 1 annotations to the task-neutral mining contracts.

The adapter is deliberately benchmark-owned: generic mining modules do not import the
competition schema. Raw note text remains immutable, train and holdout IDs are taken from the
frozen manifest, and only train annotations may be compiled into recognition knowledge.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from medical_kg_nlp.benchmarks.phase1.manual_gold import (
    load_phase1_directory,
    verify_manual_gold_split_manifest,
)
from medical_kg_nlp.mining.records import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    MinedDocument,
    RedistributionPolicy,
    ReviewStatus,
)
from medical_kg_nlp.mining.recognition_knowledge import RecognitionKnowledgePolicy
from medical_kg_nlp.benchmarks.phase1.ontology import PHASE1_RULE_BY_TYPE
from medical_kg_nlp.utils.io import read_source_text
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = [
    "PHASE1_MANUAL_GOLD_LABEL_SOURCE",
    "Phase1ManualGoldMiningCorpus",
    "build_phase1_reviewed_recognition_policy",
    "load_phase1_manual_gold_mining_corpus",
    "recognition_policy_to_data",
]

PHASE1_MANUAL_GOLD_LABEL_SOURCE = "phase1_manual_gold"
_Split = Literal["train", "holdout", "all"]


@dataclass(frozen=True)
class Phase1ManualGoldMiningCorpus:
    """One immutable split represented by generic mining records."""

    split: _Split
    corpus_fingerprint: str
    documents: tuple[MinedDocument, ...]
    annotations: tuple[AnnotationProposal, ...]


def load_phase1_manual_gold_mining_corpus(
    input_dir: str | Path,
    gold_dir: str | Path,
    split_manifest_path: str | Path,
    *,
    split: _Split,
) -> Phase1ManualGoldMiningCorpus:
    """Load one frozen split and validate every raw annotation offset."""

    if split not in {"train", "holdout", "all"}:
        raise ValueError(f"Unsupported Phase 1 mining split: {split!r}")
    manifest = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("Phase 1 split manifest must be an object")
    verify_manual_gold_split_manifest(dict(manifest), gold_dir, input_dir)
    corpus = manifest.get("corpus")
    if not isinstance(corpus, Mapping):
        raise ValueError("Phase 1 split manifest is missing corpus metadata")
    fingerprint = str(corpus.get("fingerprint_sha256", ""))
    if len(fingerprint) != 64:
        raise ValueError("Phase 1 corpus fingerprint must be SHA-256")

    split_by_document = _split_assignments(manifest)
    selected_ids = {
        document_id
        for document_id, assigned_split in split_by_document.items()
        if split == "all" or assigned_split == split
    }
    gold_by_document = load_phase1_directory(gold_dir)
    if selected_ids - set(gold_by_document):
        raise ValueError("Frozen Phase 1 split references missing gold files")

    documents: list[MinedDocument] = []
    annotations: list[AnnotationProposal] = []
    artifact_id = f"phase1-manual-gold:{fingerprint}"
    for source_document_id in sorted(selected_ids, key=_document_sort_key):
        assigned_split = split_by_document[source_document_id]
        text = read_source_text(Path(input_dir) / f"{source_document_id}.txt")
        document_id = f"phase1-manual-gold:{source_document_id}"
        document = MinedDocument(
            document_id=document_id,
            text=text,
            language="vi",
            note_type="clinical_note",
            source_artifact_id=artifact_id,
            access_class=AccessClass.LOCAL_PRIVATE,
            redistribution=RedistributionPolicy.PROHIBITED,
            hosted_processing_allowed=False,
            group_ids=(
                f"phase1-document:{source_document_id}",
                f"phase1-split:{assigned_split}",
            ),
            metadata={
                "source_document_id": source_document_id,
                "split": assigned_split,
                "corpus_fingerprint": fingerprint,
            },
        )
        documents.append(document)
        for row_index, raw_row in enumerate(gold_by_document[source_document_id]):
            annotation = _annotation_from_phase1_row(
                raw_row,
                row_index=row_index,
                document=document,
                source_document_id=source_document_id,
                split=assigned_split,
                corpus_fingerprint=fingerprint,
            )
            # INVARIANT: promotion evidence is rejected before inventory construction if an
            # annotation no longer addresses the immutable raw note.
            annotation.validate_offsets(document)
            annotations.append(annotation)

    return Phase1ManualGoldMiningCorpus(
        split=split,
        corpus_fingerprint=fingerprint,
        documents=tuple(documents),
        annotations=tuple(annotations),
    )


def build_phase1_reviewed_recognition_policy(
    annotation_policy: Mapping[str, Any],
    *,
    inventory_sha256: str,
) -> RecognitionKnowledgePolicy:
    """Convert reviewed strict aliases into a source-aware promotion allowlist."""

    aliases = annotation_policy.get("aliases")
    strict = aliases.get("strict") if isinstance(aliases, Mapping) else None
    if not isinstance(strict, Mapping):
        raise ValueError("Phase 1 annotation policy requires aliases.strict")
    reviewed: set[tuple[str, str]] = set()
    for source_label, raw_mentions in strict.items():
        label = str(source_label)
        if label not in PHASE1_RULE_BY_TYPE:
            raise ValueError(f"Unknown Phase 1 strict-alias type: {label!r}")
        if not isinstance(raw_mentions, Sequence) or isinstance(raw_mentions, str):
            raise ValueError(f"Strict aliases for {label!r} must be a string array")
        for raw_mention in raw_mentions:
            normalized = normalize_for_match(str(raw_mention))
            if not normalized:
                raise ValueError(f"Strict aliases for {label!r} contain an empty mention")
            reviewed.add((label, normalized))
    if not reviewed:
        raise ValueError("Phase 1 strict-alias allowlist is empty")

    return RecognitionKnowledgePolicy(
        policy_id="phase1-manual-gold-train-recognition-v2",
        accepted_inventory_sha256=(inventory_sha256,),
        source_label_types=tuple(
            sorted(
                (phase1_type, rule.internal_type)
                for phase1_type, rule in PHASE1_RULE_BY_TYPE.items()
            )
        ),
        accepted_label_sources=(PHASE1_MANUAL_GOLD_LABEL_SOURCE,),
        accepted_review_tiers=(
            "multi_document",
            "repeated_single_document",
            "singleton",
        ),
        min_occurrences=1,
        min_documents=1,
        allow_consensus_single_document=False,
        # INVARIANT: one- and two-character clinical fragments (for example ``đỏ``) are
        # never promoted as standalone recognition aliases; their surrounding phrase or
        # context gate must establish the entity boundary first.
        min_alias_characters=3,
        max_alias_characters=240,
        max_alias_tokens=40,
        max_surface_variants=20,
        # INVARIANT: a reviewed alias may add code-free type evidence beside the
        # terminology type. Contextual NER resolves the ambiguity; linking still uses
        # the original code-bearing concept and remains type constrained.
        allow_reviewed_baseline_type_conflicts=True,
        accepted_source_mentions=frozenset(reviewed),
    )


def recognition_policy_to_data(policy: RecognitionKnowledgePolicy) -> dict[str, Any]:
    """Serialize a generated policy so the exact promotion contract is auditable."""

    reviewed: dict[str, list[str]] = {}
    for source_label, mention in sorted(policy.accepted_source_mentions):
        reviewed.setdefault(source_label, []).append(mention)
    return {
        "schema_version": "mined-recognition-promotion-policy.v1",
        "policy_id": policy.policy_id,
        "accepted_inventory_sha256": list(policy.accepted_inventory_sha256),
        "source_label_types": {
            label: entity_type.value for label, entity_type in policy.source_label_types
        },
        "accepted_label_sources": list(policy.accepted_label_sources),
        "accepted_review_tiers": list(policy.accepted_review_tiers),
        "min_occurrences": policy.min_occurrences,
        "min_documents": policy.min_documents,
        "allow_consensus_single_document": policy.allow_consensus_single_document,
        "min_consensus_occurrences": policy.min_consensus_occurrences,
        "min_alias_characters": policy.min_alias_characters,
        "max_alias_characters": policy.max_alias_characters,
        "max_alias_tokens": policy.max_alias_tokens,
        "max_surface_variants": policy.max_surface_variants,
        "allow_numeric_only": policy.allow_numeric_only,
        "allow_reviewed_baseline_type_conflicts": (
            policy.allow_reviewed_baseline_type_conflicts
        ),
        "accepted_source_mentions": reviewed,
        "blocked_normalized_mentions": list(policy.blocked_normalized_mentions),
    }


def _annotation_from_phase1_row(
    raw_row: Mapping[str, Any],
    *,
    row_index: int,
    document: MinedDocument,
    source_document_id: str,
    split: str,
    corpus_fingerprint: str,
) -> AnnotationProposal:
    phase1_type = str(raw_row.get("type", ""))
    rule = PHASE1_RULE_BY_TYPE.get(phase1_type)
    if rule is None:
        raise ValueError(
            f"Phase 1 document {source_document_id} has unknown entity type {phase1_type!r}"
        )
    raw_position = raw_row.get("position")
    if not isinstance(raw_position, list) or len(raw_position) != 2:
        raise ValueError(
            f"Phase 1 document {source_document_id} row {row_index} has invalid position"
        )
    start, end = int(raw_position[0]), int(raw_position[1])
    text = str(raw_row.get("text", ""))
    identity = (
        f"{source_document_id}\0{row_index}\0{start}\0{end}\0{phase1_type}\0{text}"
    )
    return AnnotationProposal(
        annotation_id=f"phase1-gold:{hashlib.sha256(identity.encode()).hexdigest()[:24]}",
        document_id=document.document_id,
        span=(start, end),
        text=text,
        entity_type=rule.internal_type.value,
        assertions=tuple(str(value) for value in raw_row.get("assertions", [])),
        # Linking knowledge is compiled separately against pinned terminology releases. Keeping
        # concepts empty here prevents unversioned competition candidates from entering NER.
        concepts=(),
        confidence=1.0,
        layer=AnnotationLayer.GOLD,
        label_source=PHASE1_MANUAL_GOLD_LABEL_SOURCE,
        labeler_id="phase1_human_reviewed_gold",
        review_status=ReviewStatus.ACCEPTED,
        source_label=phase1_type,
        metadata={
            "source_document_id": source_document_id,
            "split": split,
            "corpus_fingerprint": corpus_fingerprint,
        },
    )


def _split_assignments(manifest: Mapping[str, Any]) -> dict[str, str]:
    raw_assignments = manifest.get("assignments")
    if not isinstance(raw_assignments, list):
        raise ValueError("Phase 1 split manifest requires assignments")
    assignments: dict[str, str] = {}
    for raw in raw_assignments:
        if not isinstance(raw, Mapping):
            raise ValueError("Phase 1 split assignment must be an object")
        document_id = str(raw.get("document_id", ""))
        split = str(raw.get("split", ""))
        if not document_id or split not in {"train", "holdout"}:
            raise ValueError("Invalid Phase 1 split assignment")
        if document_id in assignments:
            raise ValueError(f"Duplicate Phase 1 split assignment for {document_id}")
        assignments[document_id] = split
    return assignments


def _document_sort_key(document_id: str) -> tuple[int, str]:
    return (int(document_id), document_id) if document_id.isdigit() else (10**9, document_id)
