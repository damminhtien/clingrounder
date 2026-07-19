"""Mine bounded sentence co-occurrence evidence without inferring clinical causality.

The miner intentionally emits only ``CO_OCCURS_WITH``.  A same-sentence observation can
support search, review prioritization, or a later reranker, but it is not evidence that a
procedure treats a disease or that one condition causes another.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import yaml

from medical_kg_nlp.mining.policy import MiningQualityGate
from medical_kg_nlp.mining.records import (
    AnnotationLayer,
    AnnotationProposal,
    MinedDocument,
    RelationProposal,
    ReviewStatus,
)
from medical_kg_nlp.preprocessing.sentence_splitter import split_sentences
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = [
    "CooccurrenceMiningPolicy",
    "CooccurrenceMiningResult",
    "load_cooccurrence_policy",
    "mine_cooccurrence_relations",
]

_POLICY_SCHEMA_VERSION = "medical-cooccurrence-policy.v1"
_RELATION_TYPE = "CO_OCCURS_WITH"


@dataclass(frozen=True)
class CooccurrenceMiningPolicy:
    """Fail-closed source, quality, and density gates for relation observations."""

    policy_id: str
    relation_type: str
    accepted_source_ids: tuple[str, ...]
    document_metadata_filters: tuple[tuple[str, tuple[str, ...]], ...]
    accepted_layers: tuple[AnnotationLayer, ...]
    accepted_review_statuses: tuple[ReviewStatus, ...]
    allowed_entity_pairs: tuple[tuple[str, str], ...]
    require_single_concept_link: bool = True
    require_contiguous: bool = True
    require_source_text_match: bool = True
    minimum_documents: int = 2
    max_gap_characters: int = 240
    max_annotations_per_sentence: int = 24
    max_pairs_per_document: int = 300
    max_report_pairs: int = 100

    def __post_init__(self) -> None:
        if not self.policy_id.strip():
            raise ValueError("Co-occurrence policy_id must be non-empty")
        if self.relation_type != _RELATION_TYPE:
            raise ValueError(
                "Co-occurrence mining may only emit the literal CO_OCCURS_WITH relation"
            )
        if not self.accepted_source_ids:
            raise ValueError("Co-occurrence policy requires accepted_source_ids")
        if not self.accepted_layers or not self.accepted_review_statuses:
            raise ValueError("Co-occurrence annotation quality gates must be explicit")
        if not self.allowed_entity_pairs:
            raise ValueError("Co-occurrence policy requires allowed_entity_pairs")
        if any(pair != _canonical_entity_pair(*pair) for pair in self.allowed_entity_pairs):
            raise ValueError("allowed_entity_pairs must use canonical lexical order")
        if len(self.allowed_entity_pairs) != len(set(self.allowed_entity_pairs)):
            raise ValueError("allowed_entity_pairs must be unique")
        metadata_keys = [key for key, _ in self.document_metadata_filters]
        if len(metadata_keys) != len(set(metadata_keys)):
            raise ValueError("Document metadata filter keys must be unique")
        if any(
            not key.strip() or not allowed
            for key, allowed in self.document_metadata_filters
        ):
            raise ValueError("Document metadata filters require a key and allowed values")
        for name, value in (
            ("minimum_documents", self.minimum_documents),
            ("max_gap_characters", self.max_gap_characters),
            ("max_annotations_per_sentence", self.max_annotations_per_sentence),
            ("max_pairs_per_document", self.max_pairs_per_document),
            ("max_report_pairs", self.max_report_pairs),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class CooccurrenceMiningResult:
    """Deterministic relation observations plus aggregate audit metrics."""

    relations: tuple[RelationProposal, ...]
    report: dict[str, Any]


@dataclass(frozen=True, order=True)
class _SemanticNode:
    """Cross-document identity used only to calculate support and edge direction."""

    identity_kind: str
    code_system: str
    code_or_term: str
    entity_type: str


@dataclass(frozen=True)
class _Occurrence:
    document: MinedDocument
    head: AnnotationProposal
    tail: AnnotationProposal
    head_node: _SemanticNode
    tail_node: _SemanticNode
    evidence_span: tuple[int, int]
    gap_characters: int

    @property
    def semantic_pair(self) -> tuple[_SemanticNode, _SemanticNode]:
        return (self.head_node, self.tail_node)


def load_cooccurrence_policy(path: str | Path) -> CooccurrenceMiningPolicy:
    """Load the strict, versioned policy used by the sentence relation miner."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Co-occurrence policy must be an object")
    if raw.get("schema_version") != _POLICY_SCHEMA_VERSION:
        raise ValueError("Unsupported co-occurrence policy schema version")
    entity_pairs = _entity_pairs(raw.get("allowed_entity_pairs"))
    return CooccurrenceMiningPolicy(
        policy_id=_required_string(raw, "policy_id"),
        relation_type=_required_string(raw, "relation_type"),
        accepted_source_ids=_string_tuple(raw, "accepted_source_ids"),
        document_metadata_filters=tuple(
            sorted(
                (str(key), _string_values(values, f"document_metadata_filters.{key}"))
                for key, values in _optional_mapping(
                    raw, "document_metadata_filters"
                ).items()
            )
        ),
        accepted_layers=tuple(
            AnnotationLayer(value) for value in _string_tuple(raw, "accepted_layers")
        ),
        accepted_review_statuses=tuple(
            ReviewStatus(value)
            for value in _string_tuple(raw, "accepted_review_statuses")
        ),
        allowed_entity_pairs=entity_pairs,
        require_single_concept_link=bool(raw.get("require_single_concept_link", True)),
        require_contiguous=bool(raw.get("require_contiguous", True)),
        require_source_text_match=bool(raw.get("require_source_text_match", True)),
        minimum_documents=int(raw.get("minimum_documents", 2)),
        max_gap_characters=int(raw.get("max_gap_characters", 240)),
        max_annotations_per_sentence=int(
            raw.get("max_annotations_per_sentence", 24)
        ),
        max_pairs_per_document=int(raw.get("max_pairs_per_document", 300)),
        max_report_pairs=int(raw.get("max_report_pairs", 100)),
    )


def mine_cooccurrence_relations(
    documents: Sequence[MinedDocument],
    annotations: Sequence[AnnotationProposal],
    policy: CooccurrenceMiningPolicy,
    *,
    selected_document_ids: Set[str] | None = None,
) -> CooccurrenceMiningResult:
    """Mine repeatable sentence-level concept co-occurrence from selected documents.

    ``selected_document_ids`` is an optional frozen snapshot gate.  Source metadata gates in
    the policy are applied independently, so an official train/dev/test split cannot be
    bypassed by a later generic snapshot.
    """

    issues = MiningQualityGate().validate(documents, annotations)
    if issues:
        raise ValueError(
            "Cannot mine co-occurrence from invalid records:\n" + "\n".join(issues)
        )
    documents_by_id = {document.document_id: document for document in documents}
    if selected_document_ids is not None:
        unknown_ids = sorted(set(selected_document_ids) - documents_by_id.keys())
        if unknown_ids:
            raise ValueError(
                "Selected split references unknown documents: " f"{unknown_ids[:5]}"
            )

    counters: Counter[str] = Counter()
    eligible_documents: dict[str, MinedDocument] = {}
    for document in documents:
        reason = _document_rejection_reason(document, policy, selected_document_ids)
        if reason is None:
            eligible_documents[document.document_id] = document
            counters["documents_eligible"] += 1
        else:
            counters[f"documents_rejected:{reason}"] += 1

    pair_types = set(policy.allowed_entity_pairs)
    entity_types = {value for pair in pair_types for value in pair}
    annotations_by_document: dict[str, list[AnnotationProposal]] = defaultdict(list)
    semantic_nodes: dict[str, _SemanticNode] = {}
    for annotation in annotations:
        counters["annotations_seen"] += 1
        if annotation.document_id not in eligible_documents:
            counters["annotations_rejected:document_not_eligible"] += 1
            continue
        reason, semantic_node = _annotation_gate(annotation, entity_types, policy)
        if reason is not None or semantic_node is None:
            counters[f"annotations_rejected:{reason}"] += 1
            continue
        annotations_by_document[annotation.document_id].append(annotation)
        semantic_nodes[annotation.annotation_id] = semantic_node
        counters["annotations_eligible"] += 1

    occurrences: list[_Occurrence] = []
    for document_id in sorted(eligible_documents):
        document = eligible_documents[document_id]
        document_annotations = sorted(
            annotations_by_document.get(document_id, ()),
            key=lambda value: (value.span, value.entity_type, value.annotation_id),
        )
        document_occurrences = _document_occurrences(
            document,
            document_annotations,
            semantic_nodes,
            pair_types,
            policy,
            counters,
        )
        occurrences.extend(document_occurrences)

    support_documents: dict[
        tuple[_SemanticNode, _SemanticNode], set[str]
    ] = defaultdict(set)
    occurrence_counts: Counter[tuple[_SemanticNode, _SemanticNode]] = Counter()
    for occurrence in occurrences:
        support_documents[occurrence.semantic_pair].add(occurrence.document.document_id)
        occurrence_counts[occurrence.semantic_pair] += 1

    supported_pairs = {
        pair
        for pair, document_ids in support_documents.items()
        if len(document_ids) >= policy.minimum_documents
    }
    relations = tuple(
        _relation_from_occurrence(
            occurrence,
            policy,
            support_document_count=len(support_documents[occurrence.semantic_pair]),
            pair_occurrence_count=occurrence_counts[occurrence.semantic_pair],
        )
        for occurrence in sorted(
            occurrences,
            key=lambda value: (
                value.document.document_id,
                value.evidence_span,
                value.head.annotation_id,
                value.tail.annotation_id,
            ),
        )
        if occurrence.semantic_pair in supported_pairs
    )
    counters["candidate_occurrences"] = len(occurrences)
    counters["supported_semantic_pairs"] = len(supported_pairs)
    counters["emitted_relations"] = len(relations)
    counters["unsupported_semantic_pairs"] = len(support_documents) - len(
        supported_pairs
    )
    return CooccurrenceMiningResult(
        relations=relations,
        report=_build_report(
            policy,
            counters,
            support_documents,
            occurrence_counts,
            supported_pairs,
        ),
    )


def _document_occurrences(
    document: MinedDocument,
    annotations: Sequence[AnnotationProposal],
    semantic_nodes: Mapping[str, _SemanticNode],
    allowed_pairs: set[tuple[str, str]],
    policy: CooccurrenceMiningPolicy,
    counters: Counter[str],
) -> tuple[_Occurrence, ...]:
    occurrences: list[_Occurrence] = []
    assigned_annotation_ids: set[str] = set()
    for sentence in split_sentences(document.text):
        sentence_annotations = [
            annotation
            for annotation in annotations
            if sentence.span[0] <= annotation.span[0]
            and annotation.span[1] <= sentence.span[1]
        ]
        assigned_annotation_ids.update(
            annotation.annotation_id for annotation in sentence_annotations
        )
        if len(sentence_annotations) > policy.max_annotations_per_sentence:
            # SCALING: dense generated lists can create O(n^2) pairs with little useful
            # context. Skip the complete sentence instead of truncating it asymmetrically.
            counters["sentences_rejected:annotation_density"] += 1
            continue
        for left, right in combinations(sentence_annotations, 2):
            if len(occurrences) >= policy.max_pairs_per_document:
                counters["documents_truncated:pair_limit"] += 1
                return tuple(occurrences)
            entity_pair = _canonical_entity_pair(left.entity_type, right.entity_type)
            if entity_pair not in allowed_pairs:
                counters["pairs_rejected:entity_pair"] += 1
                continue
            gap = _span_gap(left.span, right.span)
            if gap is None:
                counters["pairs_rejected:overlap"] += 1
                continue
            if gap > policy.max_gap_characters:
                counters["pairs_rejected:character_gap"] += 1
                continue
            left_node = semantic_nodes[left.annotation_id]
            right_node = semantic_nodes[right.annotation_id]
            if left_node == right_node:
                counters["pairs_rejected:self_semantic_node"] += 1
                continue
            if left_node < right_node:
                head, tail = left, right
                head_node, tail_node = left_node, right_node
            else:
                head, tail = right, left
                head_node, tail_node = right_node, left_node
            occurrences.append(
                _Occurrence(
                    document=document,
                    head=head,
                    tail=tail,
                    head_node=head_node,
                    tail_node=tail_node,
                    evidence_span=sentence.span,
                    gap_characters=gap,
                )
            )
    counters["annotations_rejected:not_in_single_sentence"] += len(
        {annotation.annotation_id for annotation in annotations}
        - assigned_annotation_ids
    )
    return tuple(occurrences)


def _document_rejection_reason(
    document: MinedDocument,
    policy: CooccurrenceMiningPolicy,
    selected_document_ids: Set[str] | None,
) -> str | None:
    if selected_document_ids is not None and document.document_id not in selected_document_ids:
        return "snapshot_split"
    source_id = document.metadata.get("parser_id", "")
    if source_id not in policy.accepted_source_ids:
        return "source_id"
    if any(
        document.metadata.get(key) not in allowed_values
        for key, allowed_values in policy.document_metadata_filters
    ):
        # INVARIANT: official source splits are checked inside the miner; callers cannot
        # accidentally replace them with a later random or balanced split.
        return "metadata"
    return None


def _annotation_gate(
    annotation: AnnotationProposal,
    accepted_entity_types: set[str],
    policy: CooccurrenceMiningPolicy,
) -> tuple[str | None, _SemanticNode | None]:
    if annotation.layer not in policy.accepted_layers:
        return "layer", None
    if annotation.review_status not in policy.accepted_review_statuses:
        return "review_status", None
    if annotation.entity_type not in accepted_entity_types:
        return "entity_type", None
    if policy.require_contiguous and annotation.metadata.get("discontinuous") == "true":
        return "discontinuous", None
    if (
        policy.require_source_text_match
        and annotation.metadata.get("source_text_match") == "false"
    ):
        return "source_text_mismatch", None
    if len(annotation.concepts) > 1:
        return "ambiguous_concept_links", None
    if not annotation.concepts:
        if policy.require_single_concept_link:
            return "missing_concept_link", None
        normalized = normalize_for_match(annotation.text)
        if not normalized:
            return "empty_normalized_term", None
        return (
            None,
            _SemanticNode("term", "", normalized, annotation.entity_type),
        )
    concept = annotation.concepts[0]
    return (
        None,
        _SemanticNode(
            "concept",
            concept.code_system,
            concept.code,
            annotation.entity_type,
        ),
    )


def _relation_from_occurrence(
    occurrence: _Occurrence,
    policy: CooccurrenceMiningPolicy,
    *,
    support_document_count: int,
    pair_occurrence_count: int,
) -> RelationProposal:
    identity = "\0".join(
        (
            policy.policy_id,
            occurrence.document.document_id,
            occurrence.head.annotation_id,
            occurrence.tail.annotation_id,
            str(occurrence.evidence_span[0]),
            str(occurrence.evidence_span[1]),
        )
    )
    return RelationProposal(
        relation_id=f"cooccurrence:{hashlib.sha256(identity.encode()).hexdigest()[:24]}",
        document_id=occurrence.document.document_id,
        head_annotation_id=occurrence.head.annotation_id,
        tail_annotation_id=occurrence.tail.annotation_id,
        relation_type=policy.relation_type,
        confidence=1.0,
        layer=AnnotationLayer.BRONZE,
        label_source="sentence_cooccurrence",
        evidence_span=occurrence.evidence_span,
        labeler_id=f"{policy.policy_id}@v1",
        review_status=ReviewStatus.PROPOSED,
        metadata={
            "gap_characters": str(occurrence.gap_characters),
            "pair_occurrence_count": str(pair_occurrence_count),
            "policy_id": policy.policy_id,
            "semantic_inference": "false",
            "support_document_count": str(support_document_count),
            "symmetric": "true",
        },
    )


def _build_report(
    policy: CooccurrenceMiningPolicy,
    counters: Counter[str],
    support_documents: Mapping[tuple[_SemanticNode, _SemanticNode], set[str]],
    occurrence_counts: Mapping[tuple[_SemanticNode, _SemanticNode], int],
    supported_pairs: set[tuple[_SemanticNode, _SemanticNode]],
) -> dict[str, Any]:
    ranked_pairs = sorted(
        supported_pairs,
        key=lambda pair: (
            -len(support_documents[pair]),
            -occurrence_counts[pair],
            pair,
        ),
    )
    entity_pair_counts = Counter(
        f"{pair[0].entity_type}|{pair[1].entity_type}" for pair in supported_pairs
    )
    return {
        "schema_version": "medical-cooccurrence-report.v1",
        "policy_id": policy.policy_id,
        "relation_type": policy.relation_type,
        "semantic_contract": "same_sentence_observation_without_causal_inference",
        "counters": dict(sorted(counters.items())),
        "supported_entity_pair_counts": dict(sorted(entity_pair_counts.items())),
        "document_metadata_filters": {
            key: list(values) for key, values in policy.document_metadata_filters
        },
        "top_supported_pairs": [
            {
                "head": _semantic_node_dict(pair[0]),
                "tail": _semantic_node_dict(pair[1]),
                "document_count": len(support_documents[pair]),
                "occurrence_count": occurrence_counts[pair],
            }
            for pair in ranked_pairs[: policy.max_report_pairs]
        ],
    }


def _semantic_node_dict(node: _SemanticNode) -> dict[str, str]:
    return {
        "identity_kind": node.identity_kind,
        "code_system": node.code_system,
        "code_or_term": node.code_or_term,
        "entity_type": node.entity_type,
    }


def _span_gap(
    left: tuple[int, int], right: tuple[int, int]
) -> int | None:
    if left[1] <= right[0]:
        return right[0] - left[1]
    if right[1] <= left[0]:
        return left[0] - right[1]
    return None


def _canonical_entity_pair(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))  # type: ignore[return-value]


def _entity_pairs(value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("allowed_entity_pairs must be a non-empty list")
    pairs: list[tuple[str, str]] = []
    for index, raw_pair in enumerate(value):
        if (
            not isinstance(raw_pair, list)
            or len(raw_pair) != 2
            or not all(isinstance(item, str) and item.strip() for item in raw_pair)
        ):
            raise ValueError(f"allowed_entity_pairs[{index}] must contain two labels")
        pairs.append(_canonical_entity_pair(raw_pair[0].strip(), raw_pair[1].strip()))
    return tuple(sorted(pairs))


def _required_string(raw: Mapping[str, Any], key: str) -> str:
    value = str(raw.get(key, "")).strip()
    if not value:
        raise ValueError(f"Co-occurrence policy field {key!r} must be non-empty")
    return value


def _string_tuple(raw: Mapping[str, Any], key: str) -> tuple[str, ...]:
    return _string_values(raw.get(key), key)


def _string_values(value: Any, key: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"Co-occurrence policy field {key!r} must be a string list")
    return tuple(str(item).strip() for item in value)


def _optional_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Co-occurrence policy field {key!r} must be an object")
    return value
