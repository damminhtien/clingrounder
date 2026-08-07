"""Materialize exact terminology crosswalk evidence on mined annotations.

Crosswalk lookup and concept attachment are separate stages. Lookup may emit ambiguous
or lexical candidates for review; this module only attaches one pinned exact concept
without changing the source span, type, assertions, or review status.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from clingrounder.kg.constraints import code_system_valid_for_entity_type
from clingrounder.mining.records import AnnotationProposal, ConceptLink
from clingrounder.schema.types import CodeSystem, EntityType
from clingrounder.utils.text import normalize_for_match

__all__ = [
    "CrosswalkLinkMaterializationPolicy",
    "CrosswalkLinkMaterializationResult",
    "load_crosswalk_link_policy",
    "load_crosswalk_rows",
    "materialize_exact_crosswalk_links",
]

_POLICY_SCHEMA_VERSION = "medical-crosswalk-link-policy.v1"
_REPORT_SCHEMA_VERSION = "medical-crosswalk-link-materialization.v1"


@dataclass(frozen=True)
class CrosswalkLinkMaterializationPolicy:
    """Pinned eligibility contract for attaching review-stage concept links."""

    policy_id: str
    accepted_crosswalk_policy_ids: frozenset[str]
    accepted_candidate_sources: frozenset[str]
    accepted_code_systems: frozenset[CodeSystem]
    accepted_promotion_statuses: frozenset[str]
    append_non_conflicting_code_systems: bool = False
    confidence: float = 1.0

    def __post_init__(self) -> None:
        required = (
            self.accepted_crosswalk_policy_ids,
            self.accepted_candidate_sources,
            self.accepted_code_systems,
            self.accepted_promotion_statuses,
        )
        if not self.policy_id.strip() or any(not values for values in required):
            raise ValueError("Crosswalk-link policy fields must be explicit and non-empty")
        if CodeSystem.NONE in self.accepted_code_systems:
            raise ValueError("Crosswalk-link policy cannot attach the NONE code system")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Crosswalk-link confidence must be in [0, 1]")


@dataclass(frozen=True)
class CrosswalkLinkMaterializationResult:
    """Annotations with eligible links plus deterministic decision counts."""

    annotations: tuple[AnnotationProposal, ...]
    report: dict[str, Any]


@dataclass(frozen=True)
class _EligibleCrosswalkLink:
    source_entity_type: str
    source_label: str | None
    normalized_mention: str
    source_policy_id: str
    promotion_status: str
    match_mode: str
    candidate_source: str
    code_system: CodeSystem
    code: str

    @property
    def key(self) -> tuple[str, str | None, str]:
        return self.source_entity_type, self.source_label, self.normalized_mention


def load_crosswalk_link_policy(path: str | Path) -> CrosswalkLinkMaterializationPolicy:
    """Load a fail-closed policy for one terminology release and crosswalk."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Crosswalk-link policy must be an object")
    if raw.get("schema_version") != _POLICY_SCHEMA_VERSION:
        raise ValueError("Unsupported crosswalk-link policy schema version")
    return CrosswalkLinkMaterializationPolicy(
        policy_id=_required_string(raw, "policy_id"),
        accepted_crosswalk_policy_ids=frozenset(
            _string_list(raw, "accepted_crosswalk_policy_ids")
        ),
        accepted_candidate_sources=frozenset(
            _string_list(raw, "accepted_candidate_sources")
        ),
        accepted_code_systems=frozenset(
            CodeSystem(value) for value in _string_list(raw, "accepted_code_systems")
        ),
        accepted_promotion_statuses=frozenset(
            _string_list(raw, "accepted_promotion_statuses")
        ),
        append_non_conflicting_code_systems=_optional_bool(
            raw, "append_non_conflicting_code_systems", default=False
        ),
        confidence=float(raw.get("confidence", 1.0)),
    )


def load_crosswalk_rows(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load structured crosswalk JSONL without silently ignoring malformed rows."""

    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"{source}:{line_number}: crosswalk row must be an object")
            rows.append(raw)
    if not rows:
        raise ValueError(f"{source}: crosswalk must contain at least one row")
    return tuple(rows)


def materialize_exact_crosswalk_links(
    annotations: Sequence[AnnotationProposal],
    crosswalk_rows: Sequence[Mapping[str, Any]],
    policy: CrosswalkLinkMaterializationPolicy,
) -> CrosswalkLinkMaterializationResult:
    """Attach exact-unique review links while preserving source annotations.

    INVARIANT: this operation never changes annotation IDs, raw offsets, text,
    entity types, assertions, layers, or review statuses. Existing concept links are
    preserved and conflicting links are never overwritten.
    """

    annotation_ids = [annotation.annotation_id for annotation in annotations]
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("Cannot attach crosswalk links to duplicate annotation IDs")

    eligible: dict[tuple[str, str | None, str], _EligibleCrosswalkLink] = {}
    row_reasons: Counter[str] = Counter()
    for row in crosswalk_rows:
        link, reason = _eligible_link(row, policy)
        if link is None:
            row_reasons[reason] += 1
            continue
        existing = eligible.get(link.key)
        if existing is not None and existing != link:
            raise ValueError(f"Conflicting eligible crosswalk rows for {link.key!r}")
        if existing is not None:
            raise ValueError(f"Duplicate eligible crosswalk row for {link.key!r}")
        eligible[link.key] = link
        row_reasons["eligible"] += 1

    output: list[AnnotationProposal] = []
    annotation_reasons: Counter[str] = Counter()
    concept_counts: Counter[str] = Counter()
    for annotation in annotations:
        key = (
            annotation.entity_type,
            annotation.source_label,
            normalize_for_match(annotation.text),
        )
        link = eligible.get(key)
        if link is None:
            output.append(annotation)
            annotation_reasons["no_eligible_crosswalk"] += 1
            continue
        concept = ConceptLink(
            code_system=link.code_system.value,
            code=link.code,
            terminology_version=link.candidate_source,
            confidence=policy.confidence,
        )
        if any(_same_concept_identity(existing, concept) for existing in annotation.concepts):
            output.append(annotation)
            annotation_reasons["already_linked"] += 1
            continue
        if any(
            existing.code_system == concept.code_system
            for existing in annotation.concepts
        ):
            output.append(annotation)
            annotation_reasons["existing_code_system_conflict"] += 1
            continue
        if annotation.concepts and not policy.append_non_conflicting_code_systems:
            output.append(annotation)
            annotation_reasons["existing_concept_conflict"] += 1
            continue

        metadata = dict(annotation.metadata)
        provenance = {
            "crosswalk_link_policy_id": policy.policy_id,
            "crosswalk_source_policy_id": link.source_policy_id,
            "crosswalk_match_mode": link.match_mode,
            "crosswalk_promotion_status": link.promotion_status,
            "crosswalk_candidate_source": link.candidate_source,
        }
        for name, value in provenance.items():
            existing_value = metadata.get(name)
            if existing_value is not None and existing_value != value:
                raise ValueError(
                    f"Annotation {annotation.annotation_id!r} has conflicting {name!r}"
                )
            metadata[name] = value
        output.append(
            replace(
                annotation,
                concepts=(*annotation.concepts, concept),
                metadata=metadata,
            )
        )
        decision = (
            "linked_additional_code_system" if annotation.concepts else "linked"
        )
        annotation_reasons[decision] += 1
        concept_counts[f"{link.code_system.value}:{link.code}"] += 1

    report = {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "policy_id": policy.policy_id,
        "input_counts": {
            "annotations": len(annotations),
            "crosswalk_rows": len(crosswalk_rows),
        },
        "eligible_crosswalk_row_count": len(eligible),
        "crosswalk_row_decision_counts": dict(sorted(row_reasons.items())),
        "annotation_decision_counts": dict(sorted(annotation_reasons.items())),
        "linked_concept_counts": dict(sorted(concept_counts.items())),
        "output_annotation_count": len(output),
        "semantic_contract": "exact_terminology_review_evidence_not_clinical_gold",
    }
    return CrosswalkLinkMaterializationResult(tuple(output), report)


def _same_concept_identity(left: ConceptLink, right: ConceptLink) -> bool:
    """Compare terminology identity without treating confidence as identity."""

    return (
        left.code_system,
        left.code,
        left.terminology_version,
    ) == (
        right.code_system,
        right.code,
        right.terminology_version,
    )


def _eligible_link(
    row: Mapping[str, Any],
    policy: CrosswalkLinkMaterializationPolicy,
) -> tuple[_EligibleCrosswalkLink | None, str]:
    source_policy_id = str(row.get("policy_id") or "")
    if source_policy_id not in policy.accepted_crosswalk_policy_ids:
        return None, "crosswalk_policy_not_allowed"
    if row.get("status") != "unique_concept_exact":
        return None, f"status:{row.get('status')}"
    if row.get("match_mode") != "normalized_exact":
        return None, f"match_mode:{row.get('match_mode')}"
    promotion_status = str(row.get("promotion_status") or "")
    if promotion_status not in policy.accepted_promotion_statuses:
        return None, f"promotion_status:{promotion_status}"
    if row.get("automatic_promotion_allowed") is not False:
        return None, "automatic_promotion_not_fail_closed"
    if bool(row.get("query_truncated")):
        return None, "query_truncated"

    raw_candidates = row.get("candidates")
    if (
        not isinstance(raw_candidates, list)
        or len(raw_candidates) != 1
        or int(row.get("candidate_count", -1)) != 1
        or int(row.get("code_count", -1)) != 1
    ):
        return None, "candidate_not_exact_unique"
    candidate = raw_candidates[0]
    if not isinstance(candidate, Mapping):
        return None, "candidate_not_object"
    candidate_source = str(candidate.get("source") or "")
    if candidate_source not in policy.accepted_candidate_sources:
        return None, "candidate_source_not_allowed"
    try:
        code_system = CodeSystem(str(candidate.get("code_system")))
        source_entity_type = EntityType(str(row.get("source_entity_type")))
        candidate_entity_type = EntityType(str(candidate.get("semantic_type")))
    except ValueError:
        return None, "unknown_type_or_code_system"
    if code_system not in policy.accepted_code_systems:
        return None, "code_system_not_allowed"
    if not code_system_valid_for_entity_type(source_entity_type, code_system):
        return None, "code_system_invalid_for_source_type"
    if not code_system_valid_for_entity_type(candidate_entity_type, code_system):
        return None, "code_system_invalid_for_candidate_type"

    normalized_mention = str(row.get("normalized_mention") or "")
    code = str(candidate.get("code") or "")
    if not normalized_mention or normalized_mention != normalize_for_match(normalized_mention):
        return None, "invalid_normalized_mention"
    if not code:
        return None, "missing_code"
    source_label = row.get("source_label")
    return (
        _EligibleCrosswalkLink(
            source_entity_type=source_entity_type.value,
            source_label=None if source_label is None else str(source_label),
            normalized_mention=normalized_mention,
            source_policy_id=source_policy_id,
            promotion_status=promotion_status,
            match_mode="normalized_exact",
            candidate_source=candidate_source,
            code_system=code_system,
            code=code,
        ),
        "eligible",
    )


def _required_string(raw: Mapping[str, Any], field_name: str) -> str:
    value = str(raw.get(field_name) or "")
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _string_list(raw: Mapping[str, Any], field_name: str) -> tuple[str, ...]:
    value = raw.get(field_name)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list")
    output = tuple(str(item) for item in value)
    if any(not item.strip() for item in output):
        raise ValueError(f"{field_name} must contain non-empty strings")
    return output


def _optional_bool(
    raw: Mapping[str, Any], field_name: str, *, default: bool
) -> bool:
    value = raw.get(field_name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value
