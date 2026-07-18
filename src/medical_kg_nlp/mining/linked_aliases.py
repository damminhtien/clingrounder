"""Build terminology alias proposals from source annotations with explicit concept links."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from medical_kg_nlp.mining.policy import MiningQualityGate
from medical_kg_nlp.mining.records import (
    AnnotationProposal,
    MinedDocument,
    SourceArtifact,
)
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = [
    "LinkedAliasProposalPolicy",
    "LinkedAliasProposalResult",
    "build_linked_alias_proposals",
    "load_linked_alias_policy",
]

_POLICY_SCHEMA_VERSION = "linked-alias-proposal-policy.v1"


@dataclass(frozen=True)
class LinkedAliasProposalPolicy:
    """Source-pinned gates for reusing human concept-linked spans as aliases."""

    policy_id: str
    accepted_source_ids: tuple[str, ...]
    accepted_source_sha256: tuple[str, ...]
    source_code_systems: tuple[tuple[str, CodeSystem], ...]
    source_label_types: tuple[tuple[str, EntityType], ...]
    accepted_label_sources: tuple[str, ...]
    accepted_layers: tuple[str, ...]
    accepted_review_statuses: tuple[str, ...]
    proposal_review_status: str
    alias_tty: str
    document_metadata_filters: tuple[tuple[str, tuple[str, ...]], ...] = ()
    min_occurrences: int = 2
    min_documents: int = 2
    require_contiguous: bool = True
    require_source_text_match: bool = True
    max_annotation_examples: int = 20

    def __post_init__(self) -> None:
        required_collections = (
            self.accepted_source_ids,
            self.accepted_source_sha256,
            self.source_code_systems,
            self.source_label_types,
            self.accepted_label_sources,
            self.accepted_layers,
            self.accepted_review_statuses,
        )
        if not self.policy_id.strip() or any(not value for value in required_collections):
            raise ValueError("Linked-alias policy fields must be explicit and non-empty")
        if not self.proposal_review_status.strip() or not self.alias_tty.strip():
            raise ValueError("Linked-alias proposal status and TTY must be non-empty")
        if self.min_occurrences < 1 or self.min_documents < 1:
            raise ValueError("Linked-alias support thresholds must be positive")
        if self.max_annotation_examples < 1:
            raise ValueError("max_annotation_examples must be positive")
        for values, name in (
            (self.source_code_systems, "source code-system"),
            (self.source_label_types, "source-label"),
            (self.document_metadata_filters, "document-metadata"),
        ):
            keys = [key for key, _ in values]
            if len(keys) != len(set(keys)):
                raise ValueError(f"Linked-alias {name} mappings must be unique")
        if any(not key.strip() or not allowed for key, allowed in self.document_metadata_filters):
            raise ValueError("Document metadata filters require a key and allowed values")


@dataclass(frozen=True)
class LinkedAliasProposalResult:
    """Alias proposals, row-level decisions, and aggregate source audit metrics."""

    proposals: tuple[dict[str, Any], ...]
    decisions: tuple[dict[str, Any], ...]
    report: dict[str, Any]


@dataclass(frozen=True)
class _LinkedMention:
    annotation: AnnotationProposal
    document: MinedDocument
    artifact: SourceArtifact
    normalized_alias: str
    code_system: CodeSystem
    code: str
    entity_type: EntityType


def load_linked_alias_policy(path: str | Path) -> LinkedAliasProposalPolicy:
    """Load a versioned source-label and code-system mapping policy."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Linked-alias policy must be an object")
    if raw.get("schema_version") != _POLICY_SCHEMA_VERSION:
        raise ValueError("Unsupported linked-alias policy schema version")
    return LinkedAliasProposalPolicy(
        policy_id=_required_string(raw, "policy_id"),
        accepted_source_ids=_string_tuple(raw, "accepted_source_ids"),
        accepted_source_sha256=_string_tuple(raw, "accepted_source_sha256"),
        source_code_systems=tuple(
            sorted(
                (str(source), CodeSystem(str(target)))
                for source, target in _mapping(raw, "source_code_systems").items()
            )
        ),
        source_label_types=tuple(
            sorted(
                (str(source), EntityType(str(target)))
                for source, target in _mapping(raw, "source_label_types").items()
            )
        ),
        accepted_label_sources=_string_tuple(raw, "accepted_label_sources"),
        accepted_layers=_string_tuple(raw, "accepted_layers"),
        accepted_review_statuses=_string_tuple(raw, "accepted_review_statuses"),
        proposal_review_status=_required_string(raw, "proposal_review_status"),
        alias_tty=_required_string(raw, "alias_tty"),
        document_metadata_filters=tuple(
            sorted(
                (str(key), _string_values(values, f"document_metadata_filters.{key}"))
                for key, values in _optional_mapping(
                    raw,
                    "document_metadata_filters",
                ).items()
            )
        ),
        min_occurrences=int(raw.get("min_occurrences", 2)),
        min_documents=int(raw.get("min_documents", 2)),
        require_contiguous=bool(raw.get("require_contiguous", True)),
        require_source_text_match=bool(raw.get("require_source_text_match", True)),
        max_annotation_examples=int(raw.get("max_annotation_examples", 20)),
    )


def build_linked_alias_proposals(
    documents: Sequence[MinedDocument],
    annotations: Sequence[AnnotationProposal],
    artifacts: Sequence[SourceArtifact],
    policy: LinkedAliasProposalPolicy,
) -> LinkedAliasProposalResult:
    """Aggregate exact source spans whose single concept link is policy-compatible."""

    issues = MiningQualityGate().validate(documents, annotations)
    if issues:
        raise ValueError("Cannot mine aliases from invalid records:\n" + "\n".join(issues))
    documents_by_id = {document.document_id: document for document in documents}
    artifacts_by_id = {artifact.artifact_id: artifact for artifact in artifacts}
    code_systems = dict(policy.source_code_systems)
    label_types = dict(policy.source_label_types)
    candidates: list[_LinkedMention] = []
    decisions: list[dict[str, Any]] = []
    for annotation in annotations:
        candidate, reason = _linked_mention(
            annotation,
            documents_by_id,
            artifacts_by_id,
            code_systems,
            label_types,
            policy,
        )
        if candidate is None:
            decisions.append(_annotation_decision(annotation, "rejected", reason))
        else:
            candidates.append(candidate)

    grouped: dict[
        tuple[str, CodeSystem, str, EntityType, str], list[_LinkedMention]
    ] = defaultdict(list)
    targets_by_alias: dict[str, set[tuple[CodeSystem, str, EntityType]]] = defaultdict(set)
    for candidate in candidates:
        artifact_sha256 = candidate.artifact.object.sha256
        key = (
            candidate.normalized_alias,
            candidate.code_system,
            candidate.code,
            candidate.entity_type,
            artifact_sha256,
        )
        grouped[key].append(candidate)
        targets_by_alias[candidate.normalized_alias].add(
            (candidate.code_system, candidate.code, candidate.entity_type)
        )

    proposals: list[dict[str, Any]] = []
    for key, mentions in sorted(grouped.items(), key=lambda item: item[0]):
        normalized_alias, code_system, code, entity_type, source_sha256 = key
        targets = targets_by_alias[normalized_alias]
        if len(targets) != 1:
            for mention in mentions:
                decisions.append(
                    _annotation_decision(
                        mention.annotation,
                        "rejected",
                        "source_target_conflict",
                    )
                )
            continue
        document_count = len({mention.document.document_id for mention in mentions})
        if len(mentions) < policy.min_occurrences:
            reason = "insufficient_occurrences"
        elif document_count < policy.min_documents:
            reason = "insufficient_document_support"
        else:
            reason = "eligible"
        if reason != "eligible":
            for mention in mentions:
                decisions.append(_annotation_decision(mention.annotation, "rejected", reason))
            continue

        artifact = mentions[0].artifact
        surfaces = Counter(mention.annotation.text for mention in mentions)
        proposal_id = _proposal_id(policy.policy_id, normalized_alias, code_system, code)
        proposals.append(
            {
                "proposal_id": proposal_id,
                "normalized_alias": normalized_alias,
                "code_system": code_system.value,
                "code": code,
                "semantic_type": entity_type.value,
                "source": artifact.source_id,
                "source_version": artifact.source_version,
                "source_sha256": source_sha256,
                "review_status": policy.proposal_review_status,
                "supporting_record_count": document_count,
                "occurrence_count": len(mentions),
                "surface_variants": [
                    {
                        "surface": surface,
                        "count": count,
                        "ttys": [policy.alias_tty],
                    }
                    for surface, count in sorted(
                        surfaces.items(),
                        key=lambda item: (-item[1], len(item[0]), item[0].casefold()),
                    )
                ],
                "source_annotation_ids": sorted(
                    mention.annotation.annotation_id for mention in mentions
                )[: policy.max_annotation_examples],
                "policy_id": policy.policy_id,
            }
        )
        for mention in mentions:
            decisions.append(
                _annotation_decision(
                    mention.annotation,
                    "proposed",
                    "supported_single_concept_link",
                    proposal_id=proposal_id,
                )
            )

    ordered_proposals = tuple(
        sorted(
            proposals,
            key=lambda row: (
                str(row["code_system"]),
                str(row["code"]),
                str(row["normalized_alias"]),
            ),
        )
    )
    ordered_decisions = tuple(
        sorted(
            decisions,
            key=lambda row: (str(row["annotation_id"]), str(row["reason"])),
        )
    )
    return LinkedAliasProposalResult(
        proposals=ordered_proposals,
        decisions=ordered_decisions,
        report=_report(policy, annotations, candidates, ordered_proposals, ordered_decisions),
    )


def _linked_mention(
    annotation: AnnotationProposal,
    documents_by_id: Mapping[str, MinedDocument],
    artifacts_by_id: Mapping[str, SourceArtifact],
    code_systems: Mapping[str, CodeSystem],
    label_types: Mapping[str, EntityType],
    policy: LinkedAliasProposalPolicy,
) -> tuple[_LinkedMention | None, str]:
    document = documents_by_id[annotation.document_id]
    artifact = artifacts_by_id.get(document.source_artifact_id)
    if artifact is None:
        return None, "unknown_source_artifact"
    if artifact.source_id not in policy.accepted_source_ids:
        return None, "source_not_allowed"
    if artifact.object.sha256 not in policy.accepted_source_sha256:
        return None, "source_fingerprint_not_allowed"
    if any(
        document.metadata.get(key) not in allowed_values
        for key, allowed_values in policy.document_metadata_filters
    ):
        # INVARIANT: train-only alias promotion is encoded in the pinned policy,
        # rather than depending on an ad hoc caller-side document filter.
        return None, "document_metadata_not_allowed"
    if annotation.label_source not in policy.accepted_label_sources:
        return None, "label_source_not_allowed"
    if annotation.layer.value not in policy.accepted_layers:
        return None, "annotation_layer_not_allowed"
    if annotation.review_status.value not in policy.accepted_review_statuses:
        return None, "review_status_not_allowed"
    if policy.require_contiguous and annotation.metadata.get("discontinuous") == "true":
        return None, "discontinuous_span"
    if policy.require_source_text_match and annotation.metadata.get("source_text_match") == "false":
        return None, "source_text_mismatch"
    if len(annotation.concepts) != 1:
        return None, "concept_link_count_not_one"
    entity_type = label_types.get(annotation.source_label or "")
    if entity_type is None:
        return None, "source_label_not_mapped"
    source_concept = annotation.concepts[0]
    code_system = code_systems.get(source_concept.code_system)
    if code_system is None or code_system is CodeSystem.NONE:
        return None, "source_code_system_not_mapped"
    normalized_alias = normalize_for_match(annotation.text)
    if not normalized_alias:
        return None, "empty_normalized_alias"
    return (
        _LinkedMention(
            annotation=annotation,
            document=document,
            artifact=artifact,
            normalized_alias=normalized_alias,
            code_system=code_system,
            code=source_concept.code.strip().upper(),
            entity_type=entity_type,
        ),
        "eligible",
    )


def _report(
    policy: LinkedAliasProposalPolicy,
    annotations: Sequence[AnnotationProposal],
    candidates: Sequence[_LinkedMention],
    proposals: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "linked-alias-proposal-report.v1",
        "policy_id": policy.policy_id,
        "annotation_count": len(annotations),
        "candidate_after_row_gate_count": len(candidates),
        "proposal_count": len(proposals),
        "proposal_occurrence_count": sum(
            int(proposal["occurrence_count"]) for proposal in proposals
        ),
        "decision_counts": dict(
            sorted(Counter(str(row["decision"]) for row in decisions).items())
        ),
        "reason_counts": dict(
            sorted(Counter(str(row["reason"]) for row in decisions).items())
        ),
        "code_system_counts": dict(
            sorted(Counter(str(row["code_system"]) for row in proposals).items())
        ),
        "entity_type_counts": dict(
            sorted(Counter(str(row["semantic_type"]) for row in proposals).items())
        ),
        "proposal_contract": (
            "source-pinned, contiguous, exact-concept-linked, multi-document"
        ),
        "document_metadata_filters": {
            key: list(values) for key, values in policy.document_metadata_filters
        },
    }


def _annotation_decision(
    annotation: AnnotationProposal,
    decision: str,
    reason: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "annotation_id": annotation.annotation_id,
        "document_id": annotation.document_id,
        "normalized_alias": normalize_for_match(annotation.text),
        "decision": decision,
        "reason": reason,
        **details,
    }


def _proposal_id(
    policy_id: str,
    normalized_alias: str,
    code_system: CodeSystem,
    code: str,
) -> str:
    identity = f"{policy_id}\0{normalized_alias}\0{code_system.value}\0{code}"
    return f"linked-alias:{hashlib.sha256(identity.encode()).hexdigest()[:24]}"


def _required_string(raw: Mapping[str, Any], key: str) -> str:
    value = str(raw.get(key, "")).strip()
    if not value:
        raise ValueError(f"Linked-alias policy field {key!r} must be non-empty")
    return value


def _string_tuple(raw: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"Linked-alias policy field {key!r} must be a string list")
    return tuple(str(item).strip() for item in value)


def _mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"Linked-alias policy field {key!r} must be an object")
    return value


def _optional_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Linked-alias policy field {key!r} must be an object")
    return value


def _string_values(value: object, key: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"Linked-alias policy field {key!r} must be a string list")
    return tuple(str(item).strip() for item in value)
