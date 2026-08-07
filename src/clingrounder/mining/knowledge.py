"""Compile mined terminology proposals into strict runtime knowledge overlays.

Mining outputs remain review evidence until an explicit policy validates source provenance,
terminology membership, semantic compatibility, and normalized-alias uniqueness.  The compiler is
source-neutral; DailyMed is the first real producer, while future ICD, LOINC, or local terminology
connectors can emit the same proposal contract.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from clingrounder.dictionaries.synonym_table import ConceptEntry
from clingrounder.schema.types import CodeSystem, EntityType
from clingrounder.terminology.ports import TerminologyRepository
from clingrounder.utils.text import normalize_for_match

__all__ = [
    "AliasKnowledgeCompilationResult",
    "MinedAliasPromotionPolicy",
    "compile_mined_aliases",
    "load_alias_promotion_policy",
]

_POLICY_SCHEMA_VERSION = "mined-alias-promotion-policy.v1"


@dataclass(frozen=True)
class MinedAliasPromotionPolicy:
    """Fail-closed policy for promoting source proposals into a derived alias index."""

    policy_id: str
    accepted_sources: tuple[str, ...]
    accepted_source_sha256: tuple[str, ...]
    accepted_review_statuses: tuple[str, ...]
    allowed_code_systems: tuple[CodeSystem, ...]
    allowed_semantic_types: tuple[EntityType, ...]
    allowed_ttys: tuple[str, ...]
    min_supporting_records: int = 1
    min_alias_characters: int = 3
    max_alias_characters: int = 240
    max_alias_tokens: int = 40
    allow_numeric_only: bool = False

    def __post_init__(self) -> None:
        if not self.policy_id.strip():
            raise ValueError("Alias promotion policy_id must be non-empty")
        if not self.accepted_sources or not self.accepted_source_sha256:
            raise ValueError("Alias promotion policy must pin source names and SHA-256 values")
        if not self.allowed_code_systems or not self.allowed_semantic_types:
            raise ValueError("Alias promotion policy must constrain code systems and entity types")
        if CodeSystem.NONE in self.allowed_code_systems:
            raise ValueError("Alias promotion policy cannot target the NONE code system")
        if self.min_supporting_records < 1:
            raise ValueError("min_supporting_records must be positive")
        if not 1 <= self.min_alias_characters <= self.max_alias_characters:
            raise ValueError("Alias character limits are invalid")
        if self.max_alias_tokens < 1:
            raise ValueError("max_alias_tokens must be positive")


@dataclass(frozen=True)
class AliasKnowledgeCompilationResult:
    """Derived overlays, compact NER concepts, and all promotion decisions."""

    alias_overlays: tuple[dict[str, Any], ...]
    recognition_concepts: tuple[dict[str, Any], ...]
    decisions: tuple[dict[str, Any], ...]
    report: dict[str, Any]


@dataclass(frozen=True)
class _AliasCandidate:
    proposal_id: str
    normalized_alias: str
    surface: str
    concept: ConceptEntry
    ttys: tuple[str, ...]
    support_count: int
    source: str
    source_version: str
    source_sha256: str


def load_alias_promotion_policy(path: str | Path) -> MinedAliasPromotionPolicy:
    """Load a versioned YAML policy with no implicit source trust."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Alias promotion policy must be an object")
    if raw.get("schema_version") != _POLICY_SCHEMA_VERSION:
        raise ValueError("Unsupported alias promotion policy schema version")
    return MinedAliasPromotionPolicy(
        policy_id=_required_string(raw, "policy_id"),
        accepted_sources=_string_tuple(raw, "accepted_sources"),
        accepted_source_sha256=_string_tuple(raw, "accepted_source_sha256"),
        accepted_review_statuses=_string_tuple(raw, "accepted_review_statuses"),
        allowed_code_systems=tuple(
            CodeSystem(value) for value in _string_tuple(raw, "allowed_code_systems")
        ),
        allowed_semantic_types=tuple(
            EntityType(value) for value in _string_tuple(raw, "allowed_semantic_types")
        ),
        allowed_ttys=_string_tuple(raw, "allowed_ttys"),
        min_supporting_records=int(raw.get("min_supporting_records", 1)),
        min_alias_characters=int(raw.get("min_alias_characters", 3)),
        max_alias_characters=int(raw.get("max_alias_characters", 240)),
        max_alias_tokens=int(raw.get("max_alias_tokens", 40)),
        allow_numeric_only=bool(raw.get("allow_numeric_only", False)),
    )


def compile_mined_aliases(
    proposals: Iterable[Mapping[str, Any]],
    repository: TerminologyRepository,
    policy: MinedAliasPromotionPolicy,
) -> AliasKnowledgeCompilationResult:
    """Promote conflict-free aliases and retain a decision for every input proposal."""

    candidates: list[_AliasCandidate] = []
    decisions: list[dict[str, Any]] = []
    concept_cache: dict[tuple[CodeSystem, str], ConceptEntry | None] = {}
    proposal_count = 0
    for raw in proposals:
        proposal_count += 1
        candidate, reason = _candidate_from_proposal(raw, repository, policy, concept_cache)
        if candidate is None:
            decisions.append(_decision(raw, decision="rejected", reason=reason))
        else:
            candidates.append(candidate)

    by_normalized: dict[str, list[_AliasCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_normalized[candidate.normalized_alias].append(candidate)

    overlays: list[dict[str, Any]] = []
    aliases_by_concept: dict[str, list[str]] = defaultdict(list)
    concepts_by_id: dict[str, ConceptEntry] = {}
    for normalized_alias, grouped in sorted(by_normalized.items()):
        target_ids = {candidate.concept.concept_id for candidate in grouped}
        if len(target_ids) != 1:
            for candidate in grouped:
                decisions.append(
                    _candidate_decision(
                        candidate,
                        decision="rejected",
                        reason="proposal_target_conflict",
                        conflicting_concept_ids=sorted(target_ids),
                    )
                )
            continue

        target = grouped[0].concept
        base_matches = repository.exact_lookup(
            normalized_alias,
            entity_type=target.semantic_type,
            code_systems=(target.code_system,),
            limit=1_000,
        )
        base_target_ids = {concept.concept_id for concept in base_matches}
        if base_target_ids - {target.concept_id}:
            for candidate in grouped:
                decisions.append(
                    _candidate_decision(
                        candidate,
                        decision="rejected",
                        reason="canonical_alias_conflict",
                        conflicting_concept_ids=sorted(base_target_ids),
                    )
                )
            continue
        if target.concept_id in base_target_ids:
            for candidate in grouped:
                decisions.append(
                    _candidate_decision(
                        candidate,
                        decision="skipped",
                        reason="already_present",
                    )
                )
            continue

        surface = min(
            {candidate.surface for candidate in grouped},
            key=_surface_sort_key,
        )
        proposal_ids = sorted({candidate.proposal_id for candidate in grouped})
        ttys = sorted({tty for candidate in grouped for tty in candidate.ttys})
        sources = sorted({candidate.source for candidate in grouped})
        source_versions = sorted({candidate.source_version for candidate in grouped})
        source_sha256 = sorted({candidate.source_sha256 for candidate in grouped})
        support_count = max(candidate.support_count for candidate in grouped)
        overlay_id = _stable_overlay_id(policy.policy_id, target.concept_id, normalized_alias)
        overlays.append(
            {
                "alias_id": overlay_id,
                "alias": surface,
                "normalized_alias": normalized_alias,
                "target_concept_id": target.concept_id,
                "code": target.code,
                "code_system": target.code_system.value,
                "semantic_type": target.semantic_type.value,
                "ttys": ttys,
                "supporting_record_count": support_count,
                "proposal_ids": proposal_ids,
                "policy_id": policy.policy_id,
                "source": "+".join(sources),
                "source_versions": source_versions,
                "source_sha256": source_sha256,
            }
        )
        aliases_by_concept[target.concept_id].append(surface)
        concepts_by_id[target.concept_id] = target
        for candidate in grouped:
            decisions.append(
                _candidate_decision(
                    candidate,
                    decision="promoted",
                    reason="unique_reviewed_alias",
                    alias_id=overlay_id,
                )
            )

    ordered_overlays = tuple(
        sorted(
            overlays,
            key=lambda row: (
                str(row["code_system"]),
                str(row["code"]),
                str(row["normalized_alias"]),
            ),
        )
    )
    recognition = tuple(
        _recognition_row(concepts_by_id[concept_id], aliases_by_concept[concept_id], policy)
        for concept_id in sorted(concepts_by_id)
    )
    ordered_decisions = tuple(
        sorted(decisions, key=lambda row: (str(row.get("proposal_id", "")), str(row["reason"])))
    )
    decision_counts = Counter(str(row["decision"]) for row in ordered_decisions)
    reason_counts = Counter(str(row["reason"]) for row in ordered_decisions)
    report: dict[str, Any] = {
        "schema_version": "mined-alias-compilation-report.v1",
        "policy_id": policy.policy_id,
        "proposal_count": proposal_count,
        "candidate_after_row_gate_count": len(candidates),
        "overlay_alias_count": len(ordered_overlays),
        "recognition_concept_count": len(recognition),
        "decision_counts": dict(sorted(decision_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "code_system_counts": dict(
            sorted(Counter(str(row["code_system"]) for row in ordered_overlays).items())
        ),
        "semantic_type_counts": dict(
            sorted(Counter(str(row["semantic_type"]) for row in ordered_overlays).items())
        ),
        "promotion_contract": (
            "source-pinned, terminology-constrained, normalized-unique, exact-runtime-overlay"
        ),
    }
    return AliasKnowledgeCompilationResult(
        alias_overlays=ordered_overlays,
        recognition_concepts=recognition,
        decisions=ordered_decisions,
        report=report,
    )


def _candidate_from_proposal(
    raw: Mapping[str, Any],
    repository: TerminologyRepository,
    policy: MinedAliasPromotionPolicy,
    concept_cache: dict[tuple[CodeSystem, str], ConceptEntry | None],
) -> tuple[_AliasCandidate | None, str]:
    try:
        proposal_id = _required_string(raw, "proposal_id")
        source = _required_string(raw, "source")
        source_sha256 = _required_string(raw, "source_sha256")
        review_status = _required_string(raw, "review_status")
        code_system = CodeSystem(_required_string(raw, "code_system"))
        code = _required_string(raw, "code")
        normalized_alias = _required_string(raw, "normalized_alias")
        support_count = int(
            raw.get(
                "supporting_record_count",
                raw.get("supporting_set_version_count", 0),
            )
        )
    except (KeyError, TypeError, ValueError):
        return None, "invalid_proposal_schema"
    if source not in policy.accepted_sources:
        return None, "source_not_allowed"
    if source_sha256 not in policy.accepted_source_sha256:
        return None, "source_fingerprint_not_allowed"
    if review_status not in policy.accepted_review_statuses:
        return None, "review_status_not_allowed"
    if code_system not in policy.allowed_code_systems:
        return None, "code_system_not_allowed"
    if support_count < policy.min_supporting_records:
        return None, "insufficient_support"
    if normalized_alias != normalize_for_match(normalized_alias):
        return None, "normalized_alias_contract_mismatch"
    if not _alias_shape_allowed(normalized_alias, policy):
        return None, "alias_shape_not_allowed"

    cache_key = (code_system, code)
    if cache_key not in concept_cache:
        concept_cache[cache_key] = repository.get_by_code(code_system, code)
    concept = concept_cache[cache_key]
    if concept is None:
        return None, "unknown_terminology_code"
    if concept.semantic_type not in policy.allowed_semantic_types:
        return None, "semantic_type_not_allowed"
    if concept.code_system != code_system or concept.code != code:
        return None, "terminology_identity_mismatch"

    surfaces, ttys, surface_reason = _eligible_surfaces(raw, normalized_alias, policy)
    if not surfaces:
        return None, surface_reason
    return (
        _AliasCandidate(
            proposal_id=proposal_id,
            normalized_alias=normalized_alias,
            surface=min(surfaces, key=_surface_sort_key),
            concept=concept,
            ttys=tuple(sorted(ttys)),
            support_count=support_count,
            source=source,
            source_version=str(raw.get("source_version", "")).strip(),
            source_sha256=source_sha256,
        ),
        "eligible",
    )


def _eligible_surfaces(
    raw: Mapping[str, Any],
    normalized_alias: str,
    policy: MinedAliasPromotionPolicy,
) -> tuple[set[str], set[str], str]:
    variants = raw.get("surface_variants")
    if not isinstance(variants, list) or not variants:
        return set(), set(), "missing_surface_variants"
    allowed_ttys = set(policy.allowed_ttys)
    surfaces: set[str] = set()
    observed_ttys: set[str] = set()
    for variant in variants:
        if not isinstance(variant, Mapping):
            return set(), set(), "invalid_surface_variant"
        surface = str(variant.get("surface", "")).strip()
        raw_ttys = variant.get("ttys", [])
        if not surface or not isinstance(raw_ttys, list):
            return set(), set(), "invalid_surface_variant"
        ttys = {str(value) for value in raw_ttys if str(value)}
        if allowed_ttys and not (ttys & allowed_ttys):
            continue
        if normalize_for_match(surface) != normalized_alias:
            return set(), set(), "surface_normalization_mismatch"
        surfaces.add(surface)
        observed_ttys.update(ttys)
    if not surfaces:
        return set(), set(), "tty_not_allowed"
    return surfaces, observed_ttys, "eligible"


def _alias_shape_allowed(alias: str, policy: MinedAliasPromotionPolicy) -> bool:
    if not policy.min_alias_characters <= len(alias) <= policy.max_alias_characters:
        return False
    if len(alias.split()) > policy.max_alias_tokens:
        return False
    if not policy.allow_numeric_only and not any(character.isalpha() for character in alias):
        return False
    return True


def _recognition_row(
    concept: ConceptEntry,
    aliases: list[str],
    policy: MinedAliasPromotionPolicy,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "concept_id": concept.concept_id,
        "code": concept.code,
        "code_system": concept.code_system.value,
        "canonical_name": concept.canonical_name,
        "semantic_type": concept.semantic_type.value,
        "aliases": sorted(set(aliases), key=_surface_sort_key),
        "synonyms": list(concept.synonyms),
        "abbreviations": list(concept.abbreviations),
        "parents": list(concept.parents),
        "blocked_aliases": list(concept.blocked_aliases),
        "source": f"mined_alias_compiler:{policy.policy_id}",
    }
    optional = {
        "official_name_vi": concept.official_name_vi,
        "official_name_en": concept.official_name_en,
        "parent_code": concept.parent_code,
        "rxnorm_id": concept.rxnorm_id,
        "ingredient": concept.ingredient,
        "brand_name": concept.brand_name,
        "generic_name": concept.generic_name,
        "dose_form": concept.dose_form,
        "rxnorm_tty": concept.rxnorm_tty,
        "strength": concept.strength,
    }
    row.update({key: value for key, value in optional.items() if value is not None})
    return row


def _decision(
    raw: Mapping[str, Any],
    *,
    decision: str,
    reason: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "proposal_id": str(raw.get("proposal_id", "")),
        "normalized_alias": str(raw.get("normalized_alias", "")),
        "code_system": str(raw.get("code_system", "")),
        "code": str(raw.get("code", "")),
        "decision": decision,
        "reason": reason,
        **details,
    }


def _candidate_decision(
    candidate: _AliasCandidate,
    *,
    decision: str,
    reason: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "proposal_id": candidate.proposal_id,
        "normalized_alias": candidate.normalized_alias,
        "code_system": candidate.concept.code_system.value,
        "code": candidate.concept.code,
        "target_concept_id": candidate.concept.concept_id,
        "decision": decision,
        "reason": reason,
        **details,
    }


def _required_string(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _string_tuple(raw: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list | tuple) or not value:
        raise ValueError(f"{key} must be a non-empty string array")
    output = tuple(str(item).strip() for item in value)
    if any(not item for item in output):
        raise ValueError(f"{key} contains an empty value")
    return output


def _surface_sort_key(value: str) -> tuple[int, int, str, str]:
    uppercase_penalty = int(value.isupper())
    return uppercase_penalty, len(value), value.casefold(), value


def _stable_overlay_id(policy_id: str, concept_id: str, normalized_alias: str) -> str:
    identity = f"{policy_id}\0{concept_id}\0{normalized_alias}".encode("utf-8")
    return f"mined-alias:{hashlib.sha256(identity).hexdigest()[:24]}"
