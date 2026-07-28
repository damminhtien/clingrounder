"""Task-specific features for calibrating Phase 1 entity proposals.

The feature extractor is deliberately independent from gold labels and model fitting. This keeps
the same feature contract usable for development calibration and unlabeled Round 2 inference.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from medical_kg_nlp.ner.document_structure import (
    DocumentStructure,
    DocumentStructureAnalyzer,
)
from medical_kg_nlp.ontology.phase1 import PHASE1_ALLOWED_TYPES
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = [
    "PHASE1_PROPOSAL_FEATURE_CONTRACT",
    "ProposalSourceRole",
    "extract_phase1_proposal_features",
]

PHASE1_PROPOSAL_FEATURE_CONTRACT = "phase1-proposal-features.v1"

_UNIT_RE = re.compile(
    r"(?<!\w)(?:mg|mcg|µg|g|kg|ml|l|mmol|mol|meq|iu|u|"
    r"mg/dl|mmol/l|g/l|mmhg|bpm|lần/phút)(?!\w)",
    flags=re.IGNORECASE | re.UNICODE,
)
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
_QUALITATIVE_RESULT_RE = re.compile(
    r"^(?:âm tính|dương tính|bình thường|bất thường|tăng|giảm|cao|thấp)$",
    flags=re.IGNORECASE | re.UNICODE,
)
_HASH_BUCKETS = 256


class ProposalSourceRole(StrEnum):
    """Portable source categories; concrete model/run names never become features."""

    ENSEMBLE = "ensemble"
    LLM = "llm"
    RULE = "rule"
    TOKEN_MODEL = "token_model"
    VERIFIER = "verifier"


def extract_phase1_proposal_features(
    row: Mapping[str, Any],
    source_text: str,
    source_roles: Mapping[str, ProposalSourceRole | str],
    *,
    structure: DocumentStructure | None = None,
) -> dict[str, float]:
    """Extract deterministic sparse features from one raw-offset proposal.

    INVARIANT: feature names never contain a document id or absolute offset. Lexical evidence is
    mapped into stable SHA-256 buckets so artifacts remain portable and bounded in size.
    """

    start, end, entity_type, mention = _validated_proposal(row, source_text)
    active_structure = structure or DocumentStructureAnalyzer().analyze(source_text)
    roles = _proposal_roles(row, source_roles)
    status = str(row.get("status", "unknown"))
    normalized = normalize_for_match(mention)
    words = _WORD_RE.findall(normalized)
    features: dict[str, float] = {}

    features["numeric:source_count"] = float(len(roles))
    features["numeric:source_fraction"] = len(roles) / max(1, len(source_roles))
    features["numeric:span_log_length"] = math.log1p(end - start)
    features["numeric:token_log_count"] = math.log1p(len(words))
    features[f"type:{entity_type}"] = 1.0
    features[f"status:{status}"] = 1.0
    features[f"genre:{active_structure.genre.value}"] = 1.0
    features["flag:all_source_agreement"] = float(
        bool(row.get("all_source_agreement", False))
    )

    for role in roles:
        features[f"role:{role.value}"] = 1.0
        features[f"interaction:role_type:{role.value}:{entity_type}"] = 1.0
    features[f"interaction:status_type:{status}:{entity_type}"] = 1.0
    features[
        f"interaction:genre_type:{active_structure.genre.value}:{entity_type}"
    ] = 1.0

    line = active_structure.line_at(start)
    if line is not None:
        line_length = max(1, line.span[1] - line.span[0])
        features["flag:list_item"] = float(line.is_list_item)
        features["flag:starts_list_item"] = float(
            active_structure.starts_list_item(start)
        )
        features["numeric:line_relative_start"] = (start - line.span[0]) / line_length
    section = active_structure.section_at(start)
    section_name = section.kind.value if section is not None else "none"
    features[f"section:{section_name}"] = 1.0
    features[f"interaction:section_type:{section_name}:{entity_type}"] = 1.0
    if section is not None:
        heading_start, heading_end = section.heading_span
        features["flag:inside_heading"] = float(
            start < heading_end and end > heading_start
        )

    _add_surface_features(features, mention, normalized, words)
    _add_hashed_lexical_features(features, normalized, source_text, start, end)
    return dict(sorted(features.items()))


def _validated_proposal(
    row: Mapping[str, Any],
    source_text: str,
) -> tuple[int, int, str, str]:
    position = row.get("position")
    mention = row.get("text")
    entity_type = row.get("type")
    if (
        not isinstance(position, list)
        or len(position) != 2
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in position)
        or not isinstance(mention, str)
        or not mention
        or entity_type not in PHASE1_ALLOWED_TYPES
    ):
        raise ValueError("Proposal has invalid Phase 1 span/type fields")
    start, end = position
    if start < 0 or end <= start or end > len(source_text):
        raise ValueError("Proposal position is outside source text")
    if source_text[start:end] != mention:
        raise ValueError("Proposal text does not match its raw source offset")
    return start, end, str(entity_type), mention


def _proposal_roles(
    row: Mapping[str, Any],
    source_roles: Mapping[str, ProposalSourceRole | str],
) -> tuple[ProposalSourceRole, ...]:
    raw_sources = row.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("Proposal must identify at least one source")
    roles: set[ProposalSourceRole] = set()
    for source in raw_sources:
        if not isinstance(source, str) or source not in source_roles:
            raise ValueError(f"Proposal source {source!r} has no configured role")
        try:
            roles.add(ProposalSourceRole(source_roles[source]))
        except ValueError as exc:
            raise ValueError(f"Unsupported proposal source role for {source!r}") from exc
    return tuple(sorted(roles, key=lambda role: role.value))


def _add_surface_features(
    features: dict[str, float],
    mention: str,
    normalized: str,
    words: list[str],
) -> None:
    stripped = mention.strip()
    features["flag:single_token"] = float(len(words) == 1)
    features["flag:very_short"] = float(len(normalized) <= 3)
    features["flag:short"] = float(len(normalized) <= 6)
    features["flag:contains_digit"] = float(any(char.isdigit() for char in mention))
    features["flag:contains_unit"] = float(bool(_UNIT_RE.search(normalized)))
    features["flag:contains_percent"] = float("%" in mention)
    features["flag:contains_colon"] = float(":" in mention)
    features["flag:contains_parenthesis"] = float(
        any(char in mention for char in "()[]")
    )
    features["flag:contains_slash"] = float("/" in mention)
    features["flag:starts_upper"] = float(bool(stripped and stripped[0].isupper()))
    features["flag:all_upper"] = float(
        bool(stripped) and stripped.upper() == stripped and stripped.lower() != stripped
    )
    features["flag:starts_punctuation"] = float(
        bool(stripped) and unicodedata.category(stripped[0]).startswith("P")
    )
    features["flag:ends_punctuation"] = float(
        bool(stripped) and unicodedata.category(stripped[-1]).startswith("P")
    )
    features["flag:numeric_only"] = float(
        bool(stripped) and all(char.isdigit() or char in ".,/%- " for char in stripped)
    )
    features["flag:qualitative_result"] = float(
        bool(_QUALITATIVE_RESULT_RE.fullmatch(normalized))
    )


def _add_hashed_lexical_features(
    features: dict[str, float],
    normalized: str,
    source_text: str,
    start: int,
    end: int,
) -> None:
    padded = f"^{normalized}$"
    for size in (2, 3, 4):
        for index in range(max(0, len(padded) - size + 1)):
            _add_hash(features, f"mention_char_{size}", padded[index : index + size])

    left_words = _WORD_RE.findall(normalize_for_match(source_text[max(0, start - 96) : start]))
    right_words = _WORD_RE.findall(
        normalize_for_match(source_text[end : min(len(source_text), end + 96)])
    )
    for distance, word in enumerate(reversed(left_words[-4:]), start=1):
        _add_hash(features, f"context_left_{distance}", word)
    for distance, word in enumerate(right_words[:4], start=1):
        _add_hash(features, f"context_right_{distance}", word)


def _add_hash(features: dict[str, float], namespace: str, value: str) -> None:
    digest = hashlib.sha256(f"{namespace}\0{value}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % _HASH_BUCKETS
    features[f"hash:{namespace}:{bucket:03d}"] = 1.0
