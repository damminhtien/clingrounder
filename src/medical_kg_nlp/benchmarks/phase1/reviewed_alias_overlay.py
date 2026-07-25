"""Turn reviewed Phase 1 mappings into a safe terminology overlay.

The manual-gold candidate map is useful normalization evidence, but it is not a
recognition dictionary.  This adapter deliberately keeps that boundary explicit:
it emits compiler proposals for a pinned SQLite terminology repository and leaves
NER recognition unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from medical_kg_nlp.mining.knowledge import (
    AliasKnowledgeCompilationResult,
    MinedAliasPromotionPolicy,
    compile_mined_aliases,
)
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.terminology.ports import TerminologyRepository
from medical_kg_nlp.utils.hashing import sha256_file
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = [
    "PHASE1_REVIEWED_ALIAS_SOURCE",
    "build_reviewed_alias_policy",
    "compile_reviewed_candidate_aliases",
    "load_reviewed_candidate_proposals",
    "reviewed_alias_memory_rows",
]

PHASE1_REVIEWED_ALIAS_SOURCE = "phase1_manual_gold_train"
_EXPECTED_TARGETS = {
    "CHẨN_ĐOÁN": (CodeSystem.ICD10, EntityType.DISEASE),
    "THUỐC": (CodeSystem.RXNORM, EntityType.DRUG),
}


def build_reviewed_alias_policy(source_sha256: str) -> MinedAliasPromotionPolicy:
    """Create the fail-closed policy for the current map fingerprint."""

    if len(source_sha256) != 64 or any(character not in "0123456789abcdef" for character in source_sha256):
        raise ValueError("source_sha256 must be a lowercase SHA-256 hex digest")
    return MinedAliasPromotionPolicy(
        policy_id="phase1-manual-gold-train-aliases-v1",
        accepted_sources=(PHASE1_REVIEWED_ALIAS_SOURCE,),
        accepted_source_sha256=(source_sha256,),
        accepted_review_statuses=("reviewed",),
        allowed_code_systems=(CodeSystem.ICD10, CodeSystem.RXNORM),
        allowed_semantic_types=(EntityType.DISEASE, EntityType.DRUG),
        # The map is already a reviewed code assignment. TTY is retained as
        # provenance on each proposal but is not used as an accidental filter.
        allowed_ttys=(),
        min_supporting_records=1,
        min_alias_characters=3,
        max_alias_characters=240,
        max_alias_tokens=40,
    )


def load_reviewed_candidate_proposals(
    path: str | Path,
) -> tuple[dict[str, Any], ...]:
    """Load and validate map rows before passing them to the generic compiler.

    The source map is train-only and reviewed, but it is still treated as an
    untrusted file at the boundary.  In particular, a diagnosis cannot carry a
    RxNorm code and a drug cannot carry an ICD code.
    """

    source_path = Path(path)
    source_sha256 = sha256_file(source_path)
    proposals: list[dict[str, Any]] = []
    seen_codes: dict[tuple[str, str], str] = {}
    seen_proposal_ids: set[str] = set()
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError(f"{source_path}:{line_number}: expected JSON object")
            entity_type = _required_string(raw, "entity_type", source_path, line_number)
            expected = _EXPECTED_TARGETS.get(entity_type)
            if expected is None:
                raise ValueError(
                    f"{source_path}:{line_number}: unsupported entity_type {entity_type!r}"
                )
            expected_system, expected_semantic_type = expected
            code_system = _required_string(raw, "code_system", source_path, line_number)
            if code_system != expected_system.value:
                raise ValueError(
                    f"{source_path}:{line_number}: {entity_type} requires "
                    f"{expected_system.value}, got {code_system}"
                )
            if raw.get("review_status") != "reviewed":
                raise ValueError(
                    f"{source_path}:{line_number}: only reviewed rows may be promoted"
                )
            mention = _required_string(raw, "normalized_mention", source_path, line_number)
            normalized = normalize_for_match(mention)
            if normalized != mention:
                raise ValueError(
                    f"{source_path}:{line_number}: normalized_mention is not normalized"
                )
            code = _required_string(raw, "candidate", source_path, line_number)
            proposal_id = _required_string(raw, "rule_id", source_path, line_number)
            if proposal_id in seen_proposal_ids:
                raise ValueError(
                    f"{source_path}:{line_number}: duplicate rule_id {proposal_id!r}"
                )
            seen_proposal_ids.add(proposal_id)
            mention_key = (entity_type, normalized)
            previous_code = seen_codes.get(mention_key)
            if previous_code is not None and previous_code != code:
                raise ValueError(
                    f"{source_path}:{line_number}: conflicting codes for {mention_key!r}: "
                    f"{previous_code!r} and {code!r}"
                )
            seen_codes[mention_key] = code
            support = _positive_int(
                raw.get("document_support", raw.get("occurrence_support", 0)),
                source_path,
                line_number,
            )
            stage = _required_string(raw, "candidate_stage", source_path, line_number)
            release = _required_string(raw, "dictionary_release", source_path, line_number)
            proposals.append(
                {
                    "proposal_id": proposal_id,
                    "source": PHASE1_REVIEWED_ALIAS_SOURCE,
                    "source_version": release,
                    "source_sha256": source_sha256,
                    "review_status": "reviewed",
                    "code_system": expected_system.value,
                    "code": code,
                    "normalized_alias": normalized,
                    # The map stores a normalized surface rather than raw note
                    # text. This is intentional: no document-specific offsets
                    # can leak into the runtime overlay.
                    "surface_variants": [{"surface": mention, "ttys": [stage]}],
                    "supporting_record_count": support,
                    "semantic_type": expected_semantic_type.value,
                }
            )
    if not proposals:
        raise ValueError(f"{source_path}: reviewed candidate map is empty")
    return tuple(proposals)


def compile_reviewed_candidate_aliases(
    map_path: str | Path,
    repository: TerminologyRepository,
) -> tuple[AliasKnowledgeCompilationResult, str]:
    """Compile the map against a repository and return its pinned source SHA."""

    source_sha256 = sha256_file(map_path)
    proposals = load_reviewed_candidate_proposals(map_path)
    # INVARIANT: the compiler validates every target code in the loaded
    # terminology repository before an alias can be promoted.
    result = compile_mined_aliases(
        proposals,
        repository,
        build_reviewed_alias_policy(source_sha256),
    )
    return result, source_sha256


def reviewed_alias_memory_rows(
    result: AliasKnowledgeCompilationResult,
) -> tuple[dict[str, Any], ...]:
    """Return terminal exact mappings for aliases that passed promotion.

    The memory file is derived from promoted overlays rather than the input map,
    so rejected conflicts and unknown codes cannot bypass the compiler through a
    faster retrieval path.
    """

    rows = [
        {
            "mention": str(alias["alias"]),
            "entity_type": str(alias["semantic_type"]),
            "code_system": str(alias["code_system"]),
            "code": str(alias["code"]),
            "provenance": f"reviewed_memory:{alias['policy_id']}",
            "review_status": "reviewed",
            "source_sha256": list(alias["source_sha256"]),
            "source_versions": list(alias["source_versions"]),
        }
        for alias in result.alias_overlays
    ]
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                normalize_for_match(row["mention"]),
                row["code_system"],
                row["code"],
            ),
        )
    )


def _required_string(
    raw: Mapping[str, Any],
    key: str,
    path: Path,
    line_number: int,
) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{line_number}: {key} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, path: Path, line_number: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{path}:{line_number}: support count must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{line_number}: support count must be an integer") from exc
    if parsed < 1:
        raise ValueError(f"{path}:{line_number}: support count must be positive")
    return parsed
