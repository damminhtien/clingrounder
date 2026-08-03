"""Task-specific features for calibrating Phase 1 entity proposals.

The feature extractor is deliberately independent from gold labels and model fitting. This keeps
the same feature contract usable for development calibration and unlabeled Round 2 inference.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from medical_kg_nlp.ner.document_structure import (
    DocumentGenre,
    DocumentStructure,
    DocumentStructureAnalyzer,
    classify_section_heading_label,
)
from medical_kg_nlp.benchmarks.phase1.ontology import PHASE1_ALLOWED_TYPES
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = [
    "PHASE1_PROPOSAL_FEATURE_CONTRACT",
    "Phase1GenreBucket",
    "Phase1ProposalContext",
    "ProposalSourceRole",
    "extract_phase1_proposal_context",
    "extract_phase1_proposal_features",
    "is_phase1_heading_only_proposal",
    "phase1_genre_bucket",
]

PHASE1_PROPOSAL_FEATURE_CONTRACT = "phase1-proposal-features.v5"

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
_CONTEXT_WINDOW = 96
_QUESTION_RE = re.compile(
    r"^[ \t]*(?:câu[ \t]+hỏi|hỏi|question)[ \t]*(?::|-)",
    flags=re.IGNORECASE | re.UNICODE,
)
_ANSWER_RE = re.compile(
    r"^[ \t]*(?:đáp[ \t]+án|trả[ \t]+lời|answer)[ \t]*(?::|-)",
    flags=re.IGNORECASE | re.UNICODE,
)


class ProposalSourceRole(StrEnum):
    """Portable source categories; concrete model/run names never become features."""

    ENSEMBLE = "ensemble"
    LLM = "llm"
    RULE = "rule"
    TOKEN_MODEL = "token_model"
    VERIFIER = "verifier"


class Phase1GenreBucket(StrEnum):
    """Calibration buckets that share one encoder but may use separate operating points."""

    CLINICAL = "clinical"
    EDUCATIONAL = "educational"
    QUESTION_ANSWER = "qa"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Phase1ProposalContext:
    """Inspectable structural context stored alongside hashed model features."""

    left_context: str
    right_context: str
    section: str
    genre: str
    question_answer_role: str


def phase1_genre_bucket(genre: DocumentGenre | str) -> Phase1GenreBucket:
    """Collapse structural genres into the three task regimes plus a safe fallback."""

    normalized = DocumentGenre(genre)
    if normalized in {
        DocumentGenre.CLINICAL_NOTE,
        DocumentGenre.LAB_TABLE,
        DocumentGenre.MEDICATION_LIST,
    }:
        return Phase1GenreBucket.CLINICAL
    if normalized is DocumentGenre.QUESTION_ANSWER:
        return Phase1GenreBucket.QUESTION_ANSWER
    if normalized is DocumentGenre.EDUCATIONAL:
        return Phase1GenreBucket.EDUCATIONAL
    return Phase1GenreBucket.UNKNOWN


def extract_phase1_proposal_context(
    row: Mapping[str, Any],
    source_text: str,
    *,
    structure: DocumentStructure | None = None,
) -> Phase1ProposalContext:
    """Return bounded raw context without changing the proposal's source offsets."""

    start, end, _, _ = _validated_proposal(row, source_text)
    active_structure = structure or DocumentStructureAnalyzer().analyze(source_text)
    section = active_structure.section_at(start)
    return Phase1ProposalContext(
        left_context=source_text[max(0, start - _CONTEXT_WINDOW) : start],
        right_context=source_text[end : min(len(source_text), end + _CONTEXT_WINDOW)],
        section=section.kind.value if section is not None else "none",
        genre=active_structure.genre.value,
        question_answer_role=_question_answer_role(source_text, start),
    )


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
    sources = _proposal_sources(row, source_roles)
    roles = tuple(sorted({role for _, role in sources}, key=lambda role: role.value))
    source_count_by_role = Counter(role for _, role in sources)
    genre = phase1_genre_bucket(active_structure.genre)
    status = str(row.get("status", "unknown"))
    normalized = normalize_for_match(mention)
    words = _WORD_RE.findall(normalized)
    features: dict[str, float] = {}
    context = extract_phase1_proposal_context(
        row,
        source_text,
        structure=active_structure,
    )

    features["numeric:source_count"] = float(len(sources))
    features["numeric:source_fraction"] = len(sources) / max(1, len(source_roles))
    features["numeric:source_role_count"] = float(len(roles))
    features["numeric:span_log_length"] = math.log1p(end - start)
    features["numeric:token_log_count"] = math.log1p(len(words))
    features[f"type:{entity_type}"] = 1.0
    features[f"status:{status}"] = 1.0
    features[f"genre:{genre.value}"] = 1.0
    features["flag:all_source_agreement"] = float(
        bool(row.get("all_source_agreement", False))
    )

    for role in roles:
        features[f"role:{role.value}"] = 1.0
        features[f"numeric:role_source_count:{role.value}"] = float(
            source_count_by_role[role]
        )
        features[f"interaction:role_type:{role.value}:{entity_type}"] = 1.0
        features[f"interaction:genre_role:{genre.value}:{role.value}"] = 1.0
        features[
            f"interaction:genre_role_type:{genre.value}:{role.value}:{entity_type}"
        ] = 1.0
    _add_source_evidence_features(features, row, sources)
    _add_conflict_features(features, row, start=start, end=end)
    features[f"interaction:status_type:{status}:{entity_type}"] = 1.0
    features[
        f"interaction:genre_type:{genre.value}:{entity_type}"
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
    section_name = context.section
    features[f"section:{section_name}"] = 1.0
    features[f"interaction:section_type:{section_name}:{entity_type}"] = 1.0
    features[f"qa_role:{context.question_answer_role}"] = 1.0
    features[
        f"interaction:qa_role_type:{context.question_answer_role}:{entity_type}"
    ] = 1.0
    if section is not None:
        heading_start, heading_end = section.heading_span
        features["flag:inside_heading"] = float(
            start < heading_end and end > heading_start
        )
    features["flag:heading_only"] = float(
        is_phase1_heading_only_proposal(
            row,
            source_text,
            structure=active_structure,
        )
    )

    _add_surface_features(features, mention, normalized, words)
    _add_hashed_lexical_features(features, normalized, source_text, start, end)
    return dict(sorted(features.items()))


def is_phase1_heading_only_proposal(
    row: Mapping[str, Any],
    source_text: str,
    *,
    structure: DocumentStructure | None = None,
) -> bool:
    """Return whether the proposal is a structural label rather than a medical entity."""

    start, end, _, mention = _validated_proposal(row, source_text)
    # Malformed source exports sometimes concatenate a numbered heading to the previous sentence.
    # Classifying the complete proposal surface catches that case without document-specific rules.
    if classify_section_heading_label(mention) is not None:
        return True
    active_structure = structure or DocumentStructureAnalyzer().analyze(source_text)
    line = active_structure.line_at(start)
    if line is None:
        return False
    owns_heading = any(
        section.heading_span[0] <= start
        and end <= section.heading_span[1]
        for section in active_structure.sections
    )
    if not owns_heading:
        return False
    delimiter = source_text.find(":", line.span[0], line.span[1])
    # INVARIANT: a section parser owns the whole physical heading line. Only the label before the
    # first delimiter is structural; a valid entity may still occur in inline content after it.
    return delimiter < 0 or end <= delimiter


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


def _proposal_sources(
    row: Mapping[str, Any],
    source_roles: Mapping[str, ProposalSourceRole | str],
) -> tuple[tuple[str, ProposalSourceRole], ...]:
    raw_sources = row.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("Proposal must identify at least one source")
    sources: list[tuple[str, ProposalSourceRole]] = []
    for source in raw_sources:
        if not isinstance(source, str) or source not in source_roles:
            raise ValueError(f"Proposal source {source!r} has no configured role")
        try:
            sources.append((source, ProposalSourceRole(source_roles[source])))
        except ValueError as exc:
            raise ValueError(f"Unsupported proposal source role for {source!r}") from exc
    if len(sources) != len({source for source, _ in sources}):
        raise ValueError("Proposal sources must be unique")
    return tuple(sorted(sources))


def _add_source_evidence_features(
    features: dict[str, float],
    row: Mapping[str, Any],
    sources: tuple[tuple[str, ProposalSourceRole], ...],
) -> None:
    raw_evidence = row.get("source_evidence", {})
    if not isinstance(raw_evidence, Mapping):
        raise ValueError("Proposal source_evidence must be an object")
    scores_by_role: dict[ProposalSourceRole, list[float]] = {}
    support_only_by_role: Counter[ProposalSourceRole] = Counter()
    label_count_by_role: Counter[ProposalSourceRole] = Counter()
    for source, role in sources:
        evidence = raw_evidence.get(source, {})
        if not isinstance(evidence, Mapping):
            raise ValueError(f"Proposal evidence for {source!r} must be an object")
        confidence = evidence.get("confidence")
        if confidence is not None:
            if (
                not isinstance(confidence, int | float)
                or isinstance(confidence, bool)
                or not math.isfinite(float(confidence))
                or not 0.0 <= float(confidence) <= 1.0
            ):
                raise ValueError(f"Proposal evidence confidence for {source!r} is invalid")
            scores_by_role.setdefault(role, []).append(float(confidence))
        if bool(evidence.get("support_only", False)):
            features[f"flag:role_support_only:{role.value}"] = 1.0
            support_only_by_role[role] += 1
        labels = evidence.get("source_labels", [])
        if not isinstance(labels, list) or not all(
            isinstance(label, str) and label for label in labels
        ):
            raise ValueError(f"Proposal source labels for {source!r} are invalid")
        for label in labels:
            _add_hash(features, f"source_label:{role.value}", label)
            label_count_by_role[role] += 1
    for role, scores in scores_by_role.items():
        # MODEL: confidence scales differ across model families. The role-specific maximum keeps
        # their evidence separate and lets logistic calibration learn the useful scale.
        features[f"flag:role_confidence_present:{role.value}"] = 1.0
        features[f"numeric:role_confidence_max:{role.value}"] = max(scores)
        features[f"numeric:role_confidence_min:{role.value}"] = min(scores)
        features[f"numeric:role_confidence_mean:{role.value}"] = sum(scores) / len(
            scores
        )
    for role, count in support_only_by_role.items():
        features[f"numeric:role_support_only_count:{role.value}"] = float(count)
    for role, count in label_count_by_role.items():
        features[f"numeric:role_source_label_count:{role.value}"] = float(count)


def _add_conflict_features(
    features: dict[str, float],
    row: Mapping[str, Any],
    *,
    start: int,
    end: int,
) -> None:
    """Describe competing boundaries/types before the learned resolver scores the row."""

    raw_overlaps = row.get("overlap_agreements", [])
    raw_type_conflicts = row.get("type_conflicts", [])
    if not isinstance(raw_overlaps, list) or not isinstance(raw_type_conflicts, list):
        raise ValueError("Proposal conflict evidence must be represented as lists")

    overlap_source_counts: list[int] = []
    contains_count = 0
    contained_by_count = 0
    left_crossing_count = 0
    right_crossing_count = 0
    for conflict in raw_overlaps:
        if not isinstance(conflict, Mapping):
            raise ValueError("Proposal overlap evidence must be an object")
        position = conflict.get("position")
        sources = conflict.get("sources", [])
        if (
            not isinstance(position, list)
            or len(position) != 2
            or not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in position
            )
            or not isinstance(sources, list)
            or not all(isinstance(source, str) and source for source in sources)
        ):
            raise ValueError("Proposal overlap evidence is malformed")
        other_start, other_end = position
        overlap_source_counts.append(len(sources))
        if start <= other_start and other_end <= end:
            contains_count += 1
        elif other_start <= start and end <= other_end:
            contained_by_count += 1
        elif other_start < start:
            left_crossing_count += 1
        else:
            right_crossing_count += 1

    conflict_source_counts: list[int] = []
    for conflict in raw_type_conflicts:
        if not isinstance(conflict, Mapping):
            raise ValueError("Proposal type-conflict evidence must be an object")
        conflict_type = conflict.get("type")
        sources = conflict.get("sources", [])
        if (
            not isinstance(conflict_type, str)
            or not conflict_type
            or not isinstance(sources, list)
            or not all(isinstance(source, str) and source for source in sources)
        ):
            raise ValueError("Proposal type-conflict evidence is malformed")
        conflict_source_counts.append(len(sources))
        features[f"conflict_type:{conflict_type}"] = 1.0

    features["numeric:overlap_count"] = float(len(raw_overlaps))
    features["numeric:type_conflict_count"] = float(len(raw_type_conflicts))
    features["numeric:contains_competitor_count"] = float(contains_count)
    features["numeric:contained_by_competitor_count"] = float(contained_by_count)
    features["numeric:left_crossing_competitor_count"] = float(left_crossing_count)
    features["numeric:right_crossing_competitor_count"] = float(right_crossing_count)
    features["flag:has_overlap_competitor"] = float(bool(raw_overlaps))
    features["flag:has_type_competitor"] = float(bool(raw_type_conflicts))
    if overlap_source_counts:
        features["numeric:max_overlap_competitor_sources"] = float(
            max(overlap_source_counts)
        )
    if conflict_source_counts:
        features["numeric:max_type_competitor_sources"] = float(
            max(conflict_source_counts)
        )


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


def _question_answer_role(source_text: str, start: int) -> str:
    line_start = source_text.rfind("\n", 0, start) + 1
    line_end = source_text.find("\n", start)
    if line_end < 0:
        line_end = len(source_text)
    line = source_text[line_start:line_end]
    if _QUESTION_RE.match(line):
        return "question"
    if _ANSWER_RE.match(line):
        return "answer"
    return "none"
