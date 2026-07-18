"""Type-gated crosswalk from mined mentions to pinned terminologies.

Exact matches are the only rows eligible for later alias-promotion review. Optional
lexical matches are evidence for a human review queue and are never promoted by this
module.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.kg.constraints import code_system_valid_for_entity_type
from medical_kg_nlp.mining.lexicon import MentionInventoryEntry
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.terminology.ports import TerminologyRepository

__all__ = [
    "MentionCrosswalkPolicy",
    "MentionCrosswalkRecord",
    "MentionCrosswalkResult",
    "crosswalk_mentions",
    "load_crosswalk_policies",
]


@dataclass(frozen=True)
class MentionCrosswalkPolicy:
    """Explicit source-label to target-terminology query contract."""

    policy_id: str
    source_entity_type: str
    source_label: str | None
    target_entity_types: tuple[EntityType, ...]
    code_systems: tuple[CodeSystem, ...]

    def __post_init__(self) -> None:
        if not self.policy_id.strip() or not self.source_entity_type.strip():
            raise ValueError("Crosswalk policy identifiers must be non-empty")
        if not self.target_entity_types or not self.code_systems:
            raise ValueError("Crosswalk policy requires target types and code systems")
        if CodeSystem.NONE in self.code_systems:
            raise ValueError("Crosswalk policy cannot query the NONE code system")
        for target_type in self.target_entity_types:
            for code_system in self.code_systems:
                if not code_system_valid_for_entity_type(target_type, code_system):
                    raise ValueError(
                        f"Crosswalk policy {self.policy_id!r} cannot map "
                        f"{target_type.value} to {code_system.value}"
                    )


@dataclass(frozen=True)
class MentionCrosswalkRecord:
    """One mined mention and its terminology lookup outcome."""

    term_id: str
    normalized_mention: str
    source_entity_type: str
    source_label: str | None
    occurrence_count: int
    document_count: int
    policy_id: str | None
    match_mode: str
    status: str
    candidate_count: int
    code_count: int
    candidates: tuple[dict[str, Any], ...]
    query_truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "term_id": self.term_id,
            "normalized_mention": self.normalized_mention,
            "source_entity_type": self.source_entity_type,
            "source_label": self.source_label,
            "occurrence_count": self.occurrence_count,
            "document_count": self.document_count,
            "policy_id": self.policy_id,
            "match_mode": self.match_mode,
            "status": self.status,
            "candidate_count": self.candidate_count,
            "code_count": self.code_count,
            "query_truncated": self.query_truncated,
            "candidates": list(self.candidates),
            "promotion_status": (
                "review_required"
                if self.status
                in {
                    "unique_concept_exact",
                    "unique_code_exact",
                    "lexical_candidates",
                }
                else "not_eligible"
            ),
            # INVARIANT: crosswalk output is review evidence, never a runtime code assignment.
            "automatic_promotion_allowed": False,
        }


@dataclass(frozen=True)
class MentionCrosswalkResult:
    """Deterministic crosswalk rows and aggregate coverage evidence."""

    records: tuple[MentionCrosswalkRecord, ...]
    report: dict[str, Any]


def load_crosswalk_policies(path: str | Path) -> tuple[MentionCrosswalkPolicy, ...]:
    """Load a strict mapping policy rather than infer source-label semantics."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Crosswalk policy must be an object")
    if raw.get("schema_version") != "medical-mention-crosswalk-policy.v1":
        raise ValueError("Unsupported crosswalk policy schema version")
    raw_policies = raw.get("policies")
    if not isinstance(raw_policies, list) or not raw_policies:
        raise ValueError("Crosswalk policy requires a non-empty policies list")
    policies = []
    for raw_policy in raw_policies:
        if not isinstance(raw_policy, Mapping):
            raise ValueError("Each crosswalk policy must be an object")
        source_label = raw_policy.get("source_label")
        policies.append(
            MentionCrosswalkPolicy(
                policy_id=str(raw_policy["policy_id"]),
                source_entity_type=str(raw_policy["source_entity_type"]),
                source_label=(
                    None if source_label is None or str(source_label) == "*" else str(source_label)
                ),
                target_entity_types=tuple(
                    EntityType(str(value))
                    for value in _string_list(
                        raw_policy.get("target_entity_types"),
                        "target_entity_types",
                    )
                ),
                code_systems=tuple(
                    CodeSystem(str(value))
                    for value in _string_list(raw_policy.get("code_systems"), "code_systems")
                ),
            )
        )
    keys = [(policy.source_entity_type, policy.source_label) for policy in policies]
    if len(keys) != len(set(keys)):
        raise ValueError("Crosswalk policy contains duplicate source selectors")
    return tuple(policies)


def crosswalk_mentions(
    entries: Sequence[MentionInventoryEntry],
    repository: TerminologyRepository,
    policies: Sequence[MentionCrosswalkPolicy],
    *,
    terminology_metadata: Mapping[str, str] | None = None,
    workers: int = 1,
    query_limit: int = 1_000,
    candidate_output_limit: int = 20,
    lexical_fallback: bool = False,
) -> MentionCrosswalkResult:
    """Resolve candidates concurrently while preserving stable result order.

    ``lexical_fallback`` widens only unmatched exact queries. Those candidates retain
    a distinct status so downstream review cannot treat them as confirmed aliases.
    """

    if workers <= 0 or query_limit <= 0 or candidate_output_limit <= 0:
        raise ValueError("Crosswalk limits and worker count must be positive")
    policy_index = _policy_index(policies)

    def resolve(entry: MentionInventoryEntry) -> MentionCrosswalkRecord:
        policy = policy_index.get((entry.entity_type, entry.source_label))
        if policy is None:
            policy = policy_index.get((entry.entity_type, None))
        if policy is None:
            return _record(entry, policy=None, status="skipped_no_policy")
        concepts: dict[str, ConceptEntry] = {}
        query_truncated = False
        for target_type in policy.target_entity_types:
            values = repository.exact_lookup(
                entry.normalized_mention,
                entity_type=target_type,
                code_systems=policy.code_systems,
                limit=query_limit,
            )
            query_truncated = query_truncated or len(values) == query_limit
            concepts.update((concept.concept_id, concept) for concept in values)
        match_mode = "normalized_exact"
        if not concepts and lexical_fallback:
            # SCALING: widening happens only after an exact miss and remains constrained by
            # target entity type, code system, and the same bounded query limit.
            match_mode = "fts_lexical"
            for target_type in policy.target_entity_types:
                values = repository.search(
                    entry.normalized_mention,
                    entity_type=target_type,
                    code_systems=policy.code_systems,
                    limit=query_limit,
                )
                query_truncated = query_truncated or len(values) == query_limit
                concepts.update((concept.concept_id, concept) for concept in values)
        ordered = tuple(
            sorted(
                concepts.values(),
                key=lambda concept: (
                    concept.code_system.value,
                    concept.code or "",
                    concept.concept_id,
                ),
            )
        )
        if match_mode == "fts_lexical" and ordered:
            status = "lexical_candidates"
        elif query_truncated:
            status = "ambiguous_truncated"
        elif not ordered:
            status = "unmatched"
        elif not _code_keys(ordered):
            status = "concept_only_exact"
        elif len(ordered) == 1:
            status = "unique_concept_exact"
        elif len(_code_keys(ordered)) == 1 and all(concept.code is not None for concept in ordered):
            status = "unique_code_exact"
        else:
            status = "ambiguous_code_exact"
        return _record(
            entry,
            policy=policy,
            match_mode=match_mode,
            status=status,
            concepts=ordered,
            query_truncated=query_truncated,
            candidate_output_limit=candidate_output_limit,
        )

    ordered_entries = tuple(sorted(entries, key=lambda entry: entry.term_id))
    if workers == 1:
        records = tuple(resolve(entry) for entry in ordered_entries)
    else:
        # SCALING: SQLite repositories use thread-local immutable connections; map
        # preserves input order while allowing independent exact lookups in parallel.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            records = tuple(executor.map(resolve, ordered_entries))
    status_counts = Counter(record.status for record in records)
    occurrence_counts: Counter[str] = Counter()
    for record in records:
        occurrence_counts[record.status] += record.occurrence_count
    unique_statuses = {"unique_concept_exact", "unique_code_exact"}
    report = {
        "schema_version": "medical-mention-crosswalk.v1",
        "entry_count": len(records),
        "occurrence_count": sum(record.occurrence_count for record in records),
        "status_entry_counts": dict(sorted(status_counts.items())),
        "status_occurrence_counts": dict(sorted(occurrence_counts.items())),
        "unique_exact_entry_count": sum(record.status in unique_statuses for record in records),
        "unique_exact_occurrence_count": sum(
            record.occurrence_count for record in records if record.status in unique_statuses
        ),
        "policy_ids": [policy.policy_id for policy in policies],
        "query": {
            "match_mode": (
                "normalized_exact_with_lexical_fallback"
                if lexical_fallback
                else "normalized_exact"
            ),
            "lexical_fallback": lexical_fallback,
            "workers": workers,
            "query_limit": query_limit,
            "candidate_output_limit": candidate_output_limit,
        },
        "terminology": dict(sorted((terminology_metadata or {}).items())),
        "promotion_policy": "review_required",
    }
    return MentionCrosswalkResult(records=records, report=report)


def _policy_index(
    policies: Sequence[MentionCrosswalkPolicy],
) -> dict[tuple[str, str | None], MentionCrosswalkPolicy]:
    result = {(policy.source_entity_type, policy.source_label): policy for policy in policies}
    if len(result) != len(policies):
        raise ValueError("Crosswalk policies contain duplicate source selectors")
    return result


def _record(
    entry: MentionInventoryEntry,
    *,
    policy: MentionCrosswalkPolicy | None,
    status: str,
    match_mode: str = "normalized_exact",
    concepts: Sequence[ConceptEntry] = (),
    query_truncated: bool = False,
    candidate_output_limit: int = 20,
) -> MentionCrosswalkRecord:
    code_keys = _code_keys(concepts)
    candidates = tuple(
        {
            "concept_id": concept.concept_id,
            "code_system": concept.code_system.value,
            "code": concept.code,
            "canonical_name": concept.canonical_name,
            "semantic_type": concept.semantic_type.value,
            "tty": concept.rxnorm_tty,
            "source": concept.source,
        }
        for concept in concepts[:candidate_output_limit]
    )
    return MentionCrosswalkRecord(
        term_id=entry.term_id,
        normalized_mention=entry.normalized_mention,
        source_entity_type=entry.entity_type,
        source_label=entry.source_label,
        occurrence_count=entry.occurrence_count,
        document_count=entry.document_count,
        policy_id=None if policy is None else policy.policy_id,
        match_mode=match_mode,
        status=status,
        candidate_count=len(concepts),
        code_count=len(code_keys),
        candidates=candidates,
        query_truncated=query_truncated,
    )


def _code_keys(concepts: Sequence[ConceptEntry]) -> set[tuple[CodeSystem, str]]:
    return {(concept.code_system, concept.code) for concept in concepts if concept.code is not None}


def _string_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list")
    result = [str(item) for item in value]
    if any(not item.strip() for item in result):
        raise ValueError(f"{field_name} must contain non-empty values")
    return result
