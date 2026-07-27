"""Compile reviewed mention inventories into code-free NER recognition concepts.

Terminology linking and entity recognition have different precision controls. This compiler may
promote source labels into local recognition concepts, but it never assigns ICD, RxNorm, or other
medical codes. Runtime linking must still query a pinned terminology repository independently.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.mining.lexicon import MentionInventoryEntry
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = [
    "RecognitionKnowledgeCompilationResult",
    "RecognitionKnowledgePolicy",
    "compile_recognition_knowledge",
    "load_recognition_knowledge_policy",
]

_POLICY_SCHEMA_VERSION = "mined-recognition-promotion-policy.v1"


@dataclass(frozen=True)
class RecognitionKnowledgePolicy:
    """Fail-closed rules for turning source labels into local NER concepts."""

    policy_id: str
    accepted_inventory_sha256: tuple[str, ...]
    source_label_types: tuple[tuple[str, EntityType], ...]
    accepted_label_sources: tuple[str, ...]
    accepted_review_tiers: tuple[str, ...]
    min_occurrences: int = 2
    min_documents: int = 2
    allow_consensus_single_document: bool = True
    min_consensus_occurrences: int = 1
    min_alias_characters: int = 3
    max_alias_characters: int = 160
    max_alias_tokens: int = 20
    max_surface_variants: int = 12
    allow_numeric_only: bool = False
    allow_reviewed_baseline_type_conflicts: bool = False
    accepted_source_mentions: frozenset[tuple[str, str]] = frozenset()
    blocked_normalized_mentions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.policy_id.strip():
            raise ValueError("Recognition policy_id must be non-empty")
        if not self.accepted_inventory_sha256:
            raise ValueError("Recognition policy must pin at least one inventory SHA-256")
        if any(len(value) != 64 for value in self.accepted_inventory_sha256):
            raise ValueError("Recognition inventory fingerprints must be SHA-256 values")
        labels = [label for label, _ in self.source_label_types]
        if not labels or len(labels) != len(set(labels)):
            raise ValueError("Recognition source-label mappings must be non-empty and unique")
        if not self.accepted_label_sources or not self.accepted_review_tiers:
            raise ValueError("Recognition policy must constrain label sources and review tiers")
        if self.min_occurrences < 1 or self.min_documents < 1:
            raise ValueError("Recognition support thresholds must be positive")
        if self.min_consensus_occurrences < 1:
            raise ValueError("min_consensus_occurrences must be positive")
        if not 1 <= self.min_alias_characters <= self.max_alias_characters:
            raise ValueError("Recognition alias character limits are invalid")
        if self.max_alias_tokens < 1 or self.max_surface_variants < 1:
            raise ValueError("Recognition alias output limits must be positive")
        if any(
            value != normalize_for_match(value)
            for value in self.blocked_normalized_mentions
        ):
            raise ValueError("Blocked mentions must use the active normalization contract")
        mapped_labels = set(labels)
        if any(label not in mapped_labels for label, _ in self.accepted_source_mentions):
            raise ValueError("Reviewed mentions must use a mapped source label")
        if any(
            not mention or mention != normalize_for_match(mention)
            for _, mention in self.accepted_source_mentions
        ):
            raise ValueError(
                "Reviewed mentions must be non-empty and use the active normalization contract"
            )

    def mapped_type(self, source_label: str | None) -> EntityType | None:
        """Return the reviewed internal type for one source-specific label."""

        if source_label is None:
            return None
        return dict(self.source_label_types).get(source_label)


@dataclass(frozen=True)
class RecognitionKnowledgeCompilationResult:
    """Code-free concepts plus one auditable decision per inventory row."""

    concepts: tuple[dict[str, Any], ...]
    decisions: tuple[dict[str, Any], ...]
    report: dict[str, Any]


@dataclass(frozen=True)
class _EligibleMention:
    entry: MentionInventoryEntry
    entity_type: EntityType


def load_recognition_knowledge_policy(path: str | Path) -> RecognitionKnowledgePolicy:
    """Load a versioned recognition policy with explicit label mappings."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Recognition policy must be an object")
    if raw.get("schema_version") != _POLICY_SCHEMA_VERSION:
        raise ValueError("Unsupported recognition policy schema version")
    raw_mappings = raw.get("source_label_types")
    if not isinstance(raw_mappings, Mapping):
        raise ValueError("Recognition policy requires a source_label_types object")
    return RecognitionKnowledgePolicy(
        policy_id=_required_string(raw, "policy_id"),
        accepted_inventory_sha256=_string_tuple(raw, "accepted_inventory_sha256"),
        source_label_types=tuple(
            sorted(
                (str(label), EntityType(str(entity_type)))
                for label, entity_type in raw_mappings.items()
            )
        ),
        accepted_label_sources=_string_tuple(raw, "accepted_label_sources"),
        accepted_review_tiers=_string_tuple(raw, "accepted_review_tiers"),
        min_occurrences=int(raw.get("min_occurrences", 2)),
        min_documents=int(raw.get("min_documents", 2)),
        allow_consensus_single_document=bool(
            raw.get("allow_consensus_single_document", True)
        ),
        min_consensus_occurrences=int(raw.get("min_consensus_occurrences", 1)),
        min_alias_characters=int(raw.get("min_alias_characters", 3)),
        max_alias_characters=int(raw.get("max_alias_characters", 160)),
        max_alias_tokens=int(raw.get("max_alias_tokens", 20)),
        max_surface_variants=int(raw.get("max_surface_variants", 12)),
        allow_numeric_only=bool(raw.get("allow_numeric_only", False)),
        allow_reviewed_baseline_type_conflicts=bool(
            raw.get("allow_reviewed_baseline_type_conflicts", False)
        ),
        accepted_source_mentions=_source_mention_allowlist(raw),
        blocked_normalized_mentions=_string_tuple(
            raw,
            "blocked_normalized_mentions",
            required=False,
        ),
    )


def compile_recognition_knowledge(
    entries: Sequence[MentionInventoryEntry],
    policy: RecognitionKnowledgePolicy,
    *,
    inventory_sha256: str,
    baseline_entries: Sequence[ConceptEntry] = (),
) -> RecognitionKnowledgeCompilationResult:
    """Promote supported, conflict-free mentions without assigning medical codes."""

    if inventory_sha256 not in policy.accepted_inventory_sha256:
        raise ValueError("Mention inventory fingerprint is not allowed by the policy")
    existing_types = _existing_alias_types(baseline_entries)
    candidates: list[_EligibleMention] = []
    decisions: list[dict[str, Any]] = []
    for entry in entries:
        candidate, reason = _eligible_mention(entry, policy)
        if candidate is None:
            decisions.append(_decision(entry, "rejected", reason))
        else:
            candidates.append(candidate)

    grouped: dict[str, list[_EligibleMention]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.entry.normalized_mention].append(candidate)

    concepts: list[dict[str, Any]] = []
    for normalized_mention, mentions in sorted(grouped.items()):
        mapped_types = {mention.entity_type for mention in mentions}
        if len(mapped_types) != 1:
            for mention in mentions:
                decisions.append(
                    _decision(
                        mention.entry,
                        "rejected",
                        "mapped_type_conflict",
                        mapped_types=sorted(value.value for value in mapped_types),
                    )
                )
            continue
        entity_type = next(iter(mapped_types))
        baseline_types = existing_types.get(normalized_mention, set())
        if entity_type in baseline_types:
            for mention in mentions:
                decisions.append(
                    _decision(mention.entry, "skipped", "already_present_same_type")
                )
            continue
        reviewed_type_conflict = (
            policy.allow_reviewed_baseline_type_conflicts
            and all(
                (
                    mention.entry.source_label,
                    normalized_mention,
                )
                in policy.accepted_source_mentions
                for mention in mentions
            )
        )
        if baseline_types and not reviewed_type_conflict:
            for mention in mentions:
                decisions.append(
                    _decision(
                        mention.entry,
                        "rejected",
                        "baseline_type_conflict",
                        baseline_types=sorted(value.value for value in baseline_types),
                    )
                )
            continue

        concept = _recognition_concept(
            normalized_mention,
            entity_type,
            mentions,
            policy,
            inventory_sha256,
        )
        concepts.append(concept)
        for mention in mentions:
            decisions.append(
                _decision(
                    mention.entry,
                    "promoted",
                    (
                        "reviewed_baseline_type_evidence"
                        if baseline_types
                        else "supported_unique_recognition_term"
                    ),
                    concept_id=concept["concept_id"],
                    mapped_type=entity_type.value,
                    baseline_types=sorted(value.value for value in baseline_types),
                )
            )

    ordered_concepts = tuple(
        sorted(
            concepts,
            key=lambda row: (str(row["semantic_type"]), str(row["canonical_name"])),
        )
    )
    ordered_decisions = tuple(
        sorted(decisions, key=lambda row: (str(row["term_id"]), str(row["reason"])))
    )
    decision_counts = Counter(str(row["decision"]) for row in ordered_decisions)
    reason_counts = Counter(str(row["reason"]) for row in ordered_decisions)
    return RecognitionKnowledgeCompilationResult(
        concepts=ordered_concepts,
        decisions=ordered_decisions,
        report={
            "schema_version": "mined-recognition-compilation-report.v1",
            "policy_id": policy.policy_id,
            "inventory_sha256": inventory_sha256,
            "inventory_entry_count": len(entries),
            "candidate_after_row_gate_count": len(candidates),
            "recognition_concept_count": len(ordered_concepts),
            "decision_counts": dict(sorted(decision_counts.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
            "entity_type_counts": dict(
                sorted(Counter(str(row["semantic_type"]) for row in ordered_concepts).items())
            ),
            "reviewed_source_mention_count": len(policy.accepted_source_mentions),
            "promotion_contract": (
                "split-frozen, inventory-pinned, source-label-mapped, code-free"
            ),
        },
    )


def _eligible_mention(
    entry: MentionInventoryEntry,
    policy: RecognitionKnowledgePolicy,
) -> tuple[_EligibleMention | None, str]:
    entity_type = policy.mapped_type(entry.source_label)
    if entity_type is None:
        return None, "source_label_not_mapped"
    observed_sources = {source for source, count in entry.label_sources if count > 0}
    if not observed_sources or not observed_sources.issubset(policy.accepted_label_sources):
        return None, "label_source_not_allowed"
    if entry.review_tier not in policy.accepted_review_tiers:
        return None, "review_tier_not_allowed"
    if entry.occurrence_count < policy.min_occurrences:
        return None, "insufficient_occurrences"
    supported_by_documents = entry.document_count >= policy.min_documents
    supported_by_consensus = (
        policy.allow_consensus_single_document
        and entry.consensus_occurrence_count >= policy.min_consensus_occurrences
    )
    if not supported_by_documents and not supported_by_consensus:
        return None, "insufficient_document_support"
    normalized = entry.normalized_mention
    if normalized != normalize_for_match(normalized):
        return None, "normalization_contract_mismatch"
    if normalized in policy.blocked_normalized_mentions:
        return None, "blocked_mention"
    # SCALING: a source-aware hash allowlist lets large inventories fail closed without
    # scanning a linear rule list for every mention.
    if policy.accepted_source_mentions and (
        entry.source_label,
        normalized,
    ) not in policy.accepted_source_mentions:
        return None, "mention_not_reviewed"
    if not _alias_shape_allowed(normalized, policy):
        return None, "alias_shape_not_allowed"
    return _EligibleMention(entry=entry, entity_type=entity_type), "eligible"


def _recognition_concept(
    normalized_mention: str,
    entity_type: EntityType,
    mentions: Sequence[_EligibleMention],
    policy: RecognitionKnowledgePolicy,
    inventory_sha256: str,
) -> dict[str, Any]:
    surfaces: Counter[str] = Counter()
    for mention in mentions:
        surfaces.update(dict(mention.entry.surface_variants))
    ordered_surfaces = sorted(
        surfaces,
        key=lambda surface: (-surfaces[surface], len(surface), surface.casefold(), surface),
    )[: policy.max_surface_variants]
    canonical_name = ordered_surfaces[0]
    identity = f"{policy.policy_id}\0{entity_type.value}\0{normalized_mention}"
    concept_id = f"MINED:{hashlib.sha256(identity.encode()).hexdigest()[:24]}"
    return {
        "concept_id": concept_id,
        "code": None,
        "code_system": CodeSystem.NONE.value,
        "canonical_name": canonical_name,
        "semantic_type": entity_type.value,
        "aliases": [surface for surface in ordered_surfaces if surface != canonical_name],
        "source": f"mined:{policy.policy_id}",
        "mining_provenance": {
            "inventory_sha256": inventory_sha256,
            "policy_id": policy.policy_id,
            "source_labels": sorted(
                {
                    mention.entry.source_label
                    for mention in mentions
                    if mention.entry.source_label is not None
                }
            ),
            "term_ids": sorted(mention.entry.term_id for mention in mentions),
            "occurrence_count": max(
                mention.entry.occurrence_count for mention in mentions
            ),
            "document_count": max(mention.entry.document_count for mention in mentions),
            "consensus_occurrence_count": max(
                mention.entry.consensus_occurrence_count for mention in mentions
            ),
        },
    }


def _existing_alias_types(
    entries: Sequence[ConceptEntry],
) -> dict[str, set[EntityType]]:
    result: dict[str, set[EntityType]] = defaultdict(set)
    for entry in entries:
        for surface in entry.all_names:
            normalized = normalize_for_match(surface)
            if normalized:
                result[normalized].add(entry.semantic_type)
    return result


def _alias_shape_allowed(
    normalized: str,
    policy: RecognitionKnowledgePolicy,
) -> bool:
    if not policy.min_alias_characters <= len(normalized) <= policy.max_alias_characters:
        return False
    if len(normalized.split()) > policy.max_alias_tokens:
        return False
    return policy.allow_numeric_only or not normalized.replace(" ", "").isdigit()


def _decision(
    entry: MentionInventoryEntry,
    decision: str,
    reason: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "term_id": entry.term_id,
        "normalized_mention": entry.normalized_mention,
        "source_label": entry.source_label,
        "decision": decision,
        "reason": reason,
        **details,
    }


def _required_string(raw: Mapping[str, Any], key: str) -> str:
    value = str(raw.get(key, "")).strip()
    if not value:
        raise ValueError(f"Recognition policy field {key!r} must be non-empty")
    return value


def _string_tuple(
    raw: Mapping[str, Any],
    key: str,
    *,
    required: bool = True,
) -> tuple[str, ...]:
    value = raw.get(key)
    if value is None and not required:
        return ()
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"Recognition policy field {key!r} must be a string list")
    return tuple(str(item).strip() for item in value)


def _source_mention_allowlist(
    raw: Mapping[str, Any],
) -> frozenset[tuple[str, str]]:
    """Decode an optional source-label keyed reviewed mention allowlist."""

    raw_allowlist = raw.get("accepted_source_mentions")
    if raw_allowlist is None:
        return frozenset()
    if not isinstance(raw_allowlist, Mapping):
        raise ValueError("accepted_source_mentions must be an object")
    values: set[tuple[str, str]] = set()
    for raw_label, raw_mentions in raw_allowlist.items():
        label = str(raw_label)
        if not isinstance(raw_mentions, Sequence) or isinstance(raw_mentions, str):
            raise ValueError(
                f"accepted_source_mentions[{label!r}] must be a string array"
            )
        values.update(
            (label, normalize_for_match(str(mention))) for mention in raw_mentions
        )
    return frozenset(values)
