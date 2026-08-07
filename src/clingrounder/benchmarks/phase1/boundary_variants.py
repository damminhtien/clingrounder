"""Generate auditable raw-offset boundary candidates around Phase 1 proposals.

The generator intentionally emits alternatives instead of deciding which span is correct. A
separate verifier learns that decision from frozen train/development labels.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from clingrounder.benchmarks.phase1.proposal_features import (
    Phase1GenreBucket,
    ProposalSourceRole,
    extract_phase1_proposal_context,
    extract_phase1_proposal_features,
    phase1_genre_bucket,
)
from clingrounder.ner.contracts import RuleNerContext
from clingrounder.ner.dictionary_matcher import DictionaryMatcher
from clingrounder.ner.document_structure import (
    DocumentStructure,
    DocumentStructureAnalyzer,
)
from clingrounder.ner.extractors.boundary import ClinicalBoundaryProposalExtractor
from clingrounder.ner.medication_mention_parser import MedicationMentionParser
from clingrounder.ner.proposal import EntityProposal
from clingrounder.benchmarks.phase1.ontology import (
    PHASE1_ALLOWED_TYPES,
    PHASE1_TYPE_BY_ENTITY_TYPE,
    PHASE1_TYPE_PRIORITY,
    PHASE1_RULE_BY_TYPE,
)
from clingrounder.utils.text import normalize_for_match

__all__ = [
    "PHASE1_BOUNDARY_FEATURE_CONTRACT",
    "BoundaryErrorLabel",
    "BoundaryGenerator",
    "Phase1BoundaryVariant",
    "boundary_cross_encoder_text",
    "extract_phase1_boundary_features",
    "generate_phase1_boundary_variants",
    "label_phase1_boundary_variant",
]

PHASE1_BOUNDARY_FEATURE_CONTRACT = "phase1-boundary-features.v2"

_WORD_RE = re.compile(r"\w+(?:[./-]\w+)*", flags=re.UNICODE)
_CLAUSE_DELIMITER_RE = re.compile(
    r"[,;:!?\n\r]|(?<!\d)\.(?!\d)|\s+(?:và|hoặc|nhưng)\s+",
    flags=re.IGNORECASE | re.UNICODE,
)
_COORDINATION_RE = re.compile(
    r"\s*(?:,|/|\bvà\b|\bhoặc\b)\s*",
    flags=re.IGNORECASE | re.UNICODE,
)
_COORDINATION_GAP_RE = re.compile(
    r"^\s*(?:,|/|\bvà\b|\bhoặc\b)?\s*$",
    flags=re.IGNORECASE | re.UNICODE,
)
_MAX_VARIANT_CHARS = 128
_MAX_VARIANT_TOKENS = 20
_MAX_TOKEN_EXTENSION = 5
_MAX_TOKEN_TRIM = 2
_CONTEXT_CHARS = 128
_HASH_BUCKETS = 512


class BoundaryErrorLabel(StrEnum):
    """Diagnostic relation between a candidate span/type and reviewed gold."""

    CORRECT = "CORRECT"
    TOO_LONG = "TOO_LONG"
    TOO_SHORT = "TOO_SHORT"
    WRONG_ENTITY = "WRONG_ENTITY"


class BoundaryGenerator(StrEnum):
    """Portable provenance categories for one span candidate."""

    CLINICAL_GRAMMAR = "clinical_grammar"
    COMPATIBLE_MODEL = "compatible_model"
    COORDINATION = "coordination"
    DICTIONARY_TRIE = "dictionary_trie"
    LLM_QUOTE = "llm_quote"
    MEDICATION_FULL_SPAN = "medication_full_span"
    MODEL_TOKEN = "model_token"
    PROPOSAL = "proposal"
    RULE_PROPOSAL = "rule_proposal"
    TOKEN_WINDOW = "token_window"


@dataclass(frozen=True, slots=True)
class Phase1BoundaryVariant:
    """One unique span/type option with merged source and generator evidence."""

    document_id: str
    variant_id: str
    family_id: str
    text: str
    entity_type: str
    position: tuple[int, int]
    sources: tuple[str, ...]
    source_evidence: tuple[
        tuple[str, float | None, tuple[str, ...], bool],
        ...,
    ]
    generators: tuple[str, ...]
    foundation_spans: tuple[tuple[int, int], ...]
    status: str
    all_source_agreement: bool

    def __post_init__(self) -> None:
        start, end = self.position
        if start < 0 or end <= start:
            raise ValueError("Boundary variant requires a non-empty raw span")
        if self.entity_type not in PHASE1_ALLOWED_TYPES:
            raise ValueError("Boundary variant has an unsupported Phase 1 type")
        if not self.sources or tuple(sorted(set(self.sources))) != self.sources:
            raise ValueError("Boundary variant sources must be non-empty and unique")
        if not self.generators or tuple(sorted(set(self.generators))) != self.generators:
            raise ValueError("Boundary variant generators must be non-empty and unique")
        if tuple(sorted(set(self.foundation_spans))) != self.foundation_spans:
            raise ValueError("Boundary variant foundation spans must be unique")

    def to_proposal_row(self) -> dict[str, Any]:
        """Project the candidate into the shared proposal feature contract."""

        return {
            "document_id": self.document_id,
            "proposal_id": self.variant_id,
            "text": self.text,
            "type": self.entity_type,
            "position": [self.position[0], self.position[1]],
            "sources": list(self.sources),
            "source_count": len(self.sources),
            "all_source_agreement": self.all_source_agreement,
            "status": self.status,
            "source_evidence": {
                source: {
                    "present": True,
                    "confidence": confidence,
                    "source_labels": list(labels),
                    "support_only": support_only,
                }
                for source, confidence, labels, support_only in self.source_evidence
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.to_proposal_row(),
            "variant_id": self.variant_id,
            "family_id": self.family_id,
            "generators": list(self.generators),
            "foundation_spans": [list(span) for span in self.foundation_spans],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Phase1BoundaryVariant:
        """Restore a persisted variant while re-running structural invariants."""

        raw_evidence = payload.get("source_evidence")
        if not isinstance(raw_evidence, Mapping):
            raise ValueError("Boundary variant source evidence must be an object")
        source_evidence: list[
            tuple[str, float | None, tuple[str, ...], bool]
        ] = []
        for source, raw_item in raw_evidence.items():
            if not isinstance(source, str) or not isinstance(raw_item, Mapping):
                raise ValueError("Boundary variant source evidence is malformed")
            raw_confidence = raw_item.get("confidence")
            confidence = (
                None
                if raw_confidence is None
                else _finite_probability(raw_confidence)
            )
            raw_labels = raw_item.get("source_labels", ())
            if not isinstance(raw_labels, Sequence) or isinstance(
                raw_labels,
                str,
            ):
                raise ValueError("Boundary variant source labels must be a list")
            source_evidence.append(
                (
                    source,
                    confidence,
                    tuple(sorted(str(label) for label in raw_labels)),
                    bool(raw_item.get("support_only", False)),
                )
            )
        return cls(
            document_id=str(payload.get("document_id", "")),
            variant_id=str(payload.get("variant_id", "")),
            family_id=str(payload.get("family_id", "")),
            text=str(payload.get("text", "")),
            entity_type=str(payload.get("type", "")),
            position=_position(payload),
            sources=_string_tuple(payload.get("sources"), "sources"),
            source_evidence=tuple(sorted(source_evidence)),
            generators=_string_tuple(payload.get("generators"), "generators"),
            foundation_spans=_span_tuple(
                payload.get("foundation_spans"),
                "foundation_spans",
            ),
            status=str(payload.get("status", "")),
            all_source_agreement=bool(
                payload.get("all_source_agreement", False)
            ),
        )


@dataclass(slots=True)
class _VariantAccumulator:
    sources: set[str]
    source_evidence: dict[str, tuple[float | None, set[str], bool]]
    generators: set[str]
    foundation_spans: set[tuple[int, int]]
    statuses: set[str]
    all_source_agreement: bool = False


def generate_phase1_boundary_variants(
    document_id: str,
    source_text: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    source_roles: Mapping[str, ProposalSourceRole | str],
    dictionary_matcher: DictionaryMatcher | None = None,
    structure: DocumentStructure | None = None,
) -> tuple[Phase1BoundaryVariant, ...]:
    """Generate bounded alternatives around existing proposals and assign overlap families.

    INVARIANT: every candidate is sliced directly from ``source_text``. No normalization or model
    tokenization is ever used to manufacture raw offsets.
    """

    active_structure = structure or DocumentStructureAnalyzer().analyze(source_text)
    foundations = tuple(
        _validated_foundation(document_id, source_text, row, source_roles)
        for row in rows
    )
    accumulators: dict[tuple[int, int, str], _VariantAccumulator] = {}
    for foundation in foundations:
        generators = {
            BoundaryGenerator.PROPOSAL.value,
            *_source_generators(foundation, source_roles),
        }
        _add_variant(
            accumulators,
            foundation["position"],
            str(foundation["type"]),
            (foundation,),
            generators,
        )

    _add_dictionary_variants(
        accumulators,
        source_text,
        foundations,
        dictionary_matcher,
    )
    _add_clinical_grammar_variants(
        accumulators,
        source_text,
        foundations,
        active_structure,
    )
    _add_medication_variants(accumulators, source_text, foundations)
    _add_token_window_variants(accumulators, source_text, foundations)
    _add_coordination_variants(accumulators, source_text, foundations)

    variants = [
        _finalize_variant(document_id, source_text, key, accumulator)
        for key, accumulator in accumulators.items()
        if _valid_variant_span(source_text, (key[0], key[1]))
    ]
    return _assign_boundary_families(document_id, variants)


def label_phase1_boundary_variant(
    variant: Phase1BoundaryVariant,
    gold_rows: Sequence[Mapping[str, Any]],
) -> BoundaryErrorLabel:
    """Assign the closest of the four boundary diagnostics without fuzzy text matching."""

    same_type: list[tuple[tuple[int, int], int]] = []
    for row in gold_rows:
        position = _position(row)
        if row.get("type") != variant.entity_type:
            continue
        if position == variant.position:
            return BoundaryErrorLabel.CORRECT
        overlap = _overlap_length(position, variant.position)
        if overlap:
            distance = abs(position[0] - variant.position[0]) + abs(
                position[1] - variant.position[1]
            )
            same_type.append((position, distance))
    if not same_type:
        return BoundaryErrorLabel.WRONG_ENTITY
    for position, _ in sorted(same_type, key=lambda item: item[1]):
        if position[0] <= variant.position[0] and variant.position[1] <= position[1]:
            return BoundaryErrorLabel.TOO_SHORT
        if variant.position[0] <= position[0] and position[1] <= variant.position[1]:
            return BoundaryErrorLabel.TOO_LONG
    return BoundaryErrorLabel.WRONG_ENTITY


def boundary_cross_encoder_text(
    variant: Phase1BoundaryVariant,
    source_text: str,
    *,
    structure: DocumentStructure | None = None,
    genre_override: Phase1GenreBucket | str | None = None,
) -> str:
    """Render the stable joint input expected by a future sequence-classification adapter."""

    active_structure = structure or DocumentStructureAnalyzer().analyze(source_text)
    context = extract_phase1_proposal_context(
        variant.to_proposal_row(),
        source_text,
        structure=active_structure,
    )
    active_genre = (
        phase1_genre_bucket(context.genre)
        if genre_override is None
        else Phase1GenreBucket(genre_override)
    )
    left = context.left_context[-_CONTEXT_CHARS:].replace("\n", " ")
    right = context.right_context[:_CONTEXT_CHARS].replace("\n", " ")
    return "\n".join(
        (
            f"[GENRE] {active_genre.value}",
            f"[SECTION] {context.section}",
            f"[QA_ROLE] {context.question_answer_role}",
            f"[TYPE] {variant.entity_type}",
            f"[LEFT] {left}",
            f"[ENTITY] {variant.text}",
            f"[RIGHT] {right}",
        )
    )


def extract_phase1_boundary_features(
    variant: Phase1BoundaryVariant,
    source_text: str,
    source_roles: Mapping[str, ProposalSourceRole | str],
    *,
    family_size: int,
    base_probability: float | None = None,
    structure: DocumentStructure | None = None,
) -> dict[str, float]:
    """Encode joint context, source evidence, and candidate/foundation geometry."""

    if family_size < 1:
        raise ValueError("Boundary family size must be positive")
    active_structure = structure or DocumentStructureAnalyzer().analyze(source_text)
    base = extract_phase1_proposal_features(
        variant.to_proposal_row(),
        source_text,
        source_roles,
        structure=active_structure,
    )
    features = {f"proposal:{name}": value for name, value in base.items()}
    genre = phase1_genre_bucket(active_structure.genre)
    features["numeric:family_log_size"] = math.log1p(family_size)
    features["numeric:foundation_log_count"] = math.log1p(
        len(variant.foundation_spans)
    )
    features["flag:original_boundary"] = float(
        variant.position in variant.foundation_spans
    )
    if base_probability is not None:
        if not 0.0 <= base_probability <= 1.0:
            raise ValueError("Base proposal probability must be within [0, 1]")
        features["numeric:base_proposal_probability"] = base_probability
    for generator in variant.generators:
        features[f"generator:{generator}"] = 1.0
        features[
            f"interaction:genre_generator:{genre.value}:{generator}"
        ] = 1.0
        features[
            f"interaction:type_generator:{variant.entity_type}:{generator}"
        ] = 1.0

    start_deltas = [variant.position[0] - start for start, _ in variant.foundation_spans]
    end_deltas = [variant.position[1] - end for _, end in variant.foundation_spans]
    features["numeric:min_abs_start_delta"] = float(
        min(abs(value) for value in start_deltas)
    )
    features["numeric:min_abs_end_delta"] = float(
        min(abs(value) for value in end_deltas)
    )
    features["flag:same_foundation_start"] = float(0 in start_deltas)
    features["flag:same_foundation_end"] = float(0 in end_deltas)
    features["flag:expands_left"] = float(any(value < 0 for value in start_deltas))
    features["flag:expands_right"] = float(any(value > 0 for value in end_deltas))
    features["flag:trims_left"] = float(any(value > 0 for value in start_deltas))
    features["flag:trims_right"] = float(any(value < 0 for value in end_deltas))

    joint_text = boundary_cross_encoder_text(
        variant,
        source_text,
        structure=active_structure,
    )
    normalized_joint = normalize_for_match(joint_text)
    words = _WORD_RE.findall(normalized_joint)
    for word in words:
        _add_hash(features, "joint_token", word)
    for first, second in zip(words, words[1:]):
        _add_hash(features, "joint_bigram", f"{first}\0{second}")
    return dict(sorted(features.items()))


def _validated_foundation(
    document_id: str,
    source_text: str,
    row: Mapping[str, Any],
    source_roles: Mapping[str, ProposalSourceRole | str],
) -> dict[str, Any]:
    position = _position(row)
    text = row.get("text")
    entity_type = row.get("type")
    raw_sources = row.get("sources")
    if (
        not isinstance(text, str)
        or source_text[position[0] : position[1]] != text
        or entity_type not in PHASE1_ALLOWED_TYPES
        or not isinstance(raw_sources, list)
        or not raw_sources
    ):
        raise ValueError("Boundary foundation has invalid raw span/type/source fields")
    for source in raw_sources:
        if not isinstance(source, str) or source not in source_roles:
            raise ValueError(f"Boundary foundation source {source!r} has no role")
    copied = dict(row)
    copied["document_id"] = document_id
    return copied


def _source_generators(
    row: Mapping[str, Any],
    source_roles: Mapping[str, ProposalSourceRole | str],
) -> set[str]:
    generators: set[str] = set()
    raw_sources = row.get("sources")
    assert isinstance(raw_sources, list)
    for source in raw_sources:
        role = ProposalSourceRole(source_roles[str(source)])
        if role is ProposalSourceRole.RULE:
            generators.add(BoundaryGenerator.RULE_PROPOSAL.value)
        elif role is ProposalSourceRole.TOKEN_MODEL:
            generators.add(BoundaryGenerator.MODEL_TOKEN.value)
        elif role in {ProposalSourceRole.LLM, ProposalSourceRole.ENSEMBLE}:
            generators.add(BoundaryGenerator.LLM_QUOTE.value)
        elif role is ProposalSourceRole.VERIFIER:
            generators.add(BoundaryGenerator.COMPATIBLE_MODEL.value)
    return generators


def _add_dictionary_variants(
    accumulators: dict[tuple[int, int, str], _VariantAccumulator],
    source_text: str,
    foundations: Sequence[Mapping[str, Any]],
    matcher: DictionaryMatcher | None,
) -> None:
    if matcher is None:
        return
    for match in matcher.find_candidates(
        source_text,
        require_boundaries=True,
        min_alias_chars=2,
    ):
        phase1_type = PHASE1_TYPE_BY_ENTITY_TYPE.get(match.entry.semantic_type)
        if phase1_type is None:
            continue
        supporting = tuple(
            row
            for row in foundations
            if row.get("type") == phase1_type
            and _overlap_length(_position(row), match.span)
        )
        if supporting:
            _add_variant(
                accumulators,
                match.span,
                phase1_type,
                supporting,
                {BoundaryGenerator.DICTIONARY_TRIE.value},
            )


def _add_clinical_grammar_variants(
    accumulators: dict[tuple[int, int, str], _VariantAccumulator],
    source_text: str,
    foundations: Sequence[Mapping[str, Any]],
    structure: DocumentStructure,
) -> None:
    proposals: list[EntityProposal] = []
    row_by_key: dict[tuple[tuple[int, int], str], Mapping[str, Any]] = {}
    for row in foundations:
        phase1_type = str(row.get("type", ""))
        internal_type = PHASE1_RULE_BY_TYPE[phase1_type].internal_type
        if phase1_type not in {"CHẨN_ĐOÁN", "TRIỆU_CHỨNG"}:
            continue
        position = _position(row)
        proposals.append(
            EntityProposal(
                span=position,
                candidate_types=(internal_type,),
                source="boundary_foundation",
                score=0.5,
            )
        )
        row_by_key[(position, phase1_type)] = row
    if not proposals:
        return
    expanded = ClinicalBoundaryProposalExtractor().propose(
        source_text,
        RuleNerContext(
            foundation_proposals=tuple(proposals),
            structure=structure,
        ),
    )
    for proposal in expanded:
        expanded_type = proposal.entity_type
        if expanded_type is None:
            continue
        expanded_phase1_type = PHASE1_TYPE_BY_ENTITY_TYPE.get(expanded_type)
        if expanded_phase1_type is None:
            continue
        foundation = next(
            (
                row
                for (span, row_type), row in row_by_key.items()
                if row_type == expanded_phase1_type
                and span[0] >= proposal.span[0]
                and span[1] <= proposal.span[1]
            ),
            None,
        )
        if foundation is not None:
            _add_variant(
                accumulators,
                proposal.span,
                expanded_phase1_type,
                (foundation,),
                {BoundaryGenerator.CLINICAL_GRAMMAR.value},
            )


def _add_medication_variants(
    accumulators: dict[tuple[int, int, str], _VariantAccumulator],
    source_text: str,
    foundations: Sequence[Mapping[str, Any]],
) -> None:
    parser = MedicationMentionParser()
    for row in foundations:
        if row.get("type") != "THUỐC":
            continue
        full_span = parser.parse(source_text, _position(row)).full_span
        if full_span != _position(row):
            _add_variant(
                accumulators,
                full_span,
                "THUỐC",
                (row,),
                {BoundaryGenerator.MEDICATION_FULL_SPAN.value},
            )


def _add_token_window_variants(
    accumulators: dict[tuple[int, int, str], _VariantAccumulator],
    source_text: str,
    foundations: Sequence[Mapping[str, Any]],
) -> None:
    for row in foundations:
        for span in _token_window_spans(source_text, _position(row)):
            if span == _position(row):
                continue
            _add_variant(
                accumulators,
                span,
                str(row.get("type", "")),
                (row,),
                {BoundaryGenerator.TOKEN_WINDOW.value},
            )


def _add_coordination_variants(
    accumulators: dict[tuple[int, int, str], _VariantAccumulator],
    source_text: str,
    foundations: Sequence[Mapping[str, Any]],
) -> None:
    for row in foundations:
        start, end = _position(row)
        mention = source_text[start:end]
        for match in _COORDINATION_RE.finditer(mention):
            for local_start, local_end in ((0, match.start()), (match.end(), len(mention))):
                span = _trim_span(source_text, (start + local_start, start + local_end))
                if span is not None:
                    _add_variant(
                        accumulators,
                        span,
                        str(row.get("type", "")),
                        (row,),
                        {BoundaryGenerator.COORDINATION.value},
                    )

    ordered = sorted(
        foundations,
        key=lambda row: (
            str(row.get("type", "")),
            _position(row),
        ),
    )
    for left, right in zip(ordered, ordered[1:]):
        if left.get("type") != right.get("type"):
            continue
        left_span = _position(left)
        right_span = _position(right)
        if left_span[1] > right_span[0] or right_span[0] - left_span[1] > 16:
            continue
        gap = source_text[left_span[1] : right_span[0]]
        if _COORDINATION_GAP_RE.fullmatch(gap):
            _add_variant(
                accumulators,
                (left_span[0], right_span[1]),
                str(left.get("type", "")),
                (left, right),
                {BoundaryGenerator.COORDINATION.value},
            )


def _token_window_spans(
    source_text: str,
    foundation: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    clause_start, clause_end = _clause_span(source_text, foundation)
    tokens = [
        (clause_start + match.start(), clause_start + match.end())
        for match in _WORD_RE.finditer(source_text[clause_start:clause_end])
    ]
    overlapping = [
        index
        for index, token_span in enumerate(tokens)
        if _overlap_length(token_span, foundation)
    ]
    if not overlapping:
        return ()
    first = min(overlapping)
    last = max(overlapping)
    spans: set[tuple[int, int]] = set()
    for left_extension in range(_MAX_TOKEN_EXTENSION + 1):
        for right_extension in range(_MAX_TOKEN_EXTENSION + 1):
            if left_extension + right_extension == 0:
                continue
            if left_extension + right_extension > _MAX_TOKEN_EXTENSION:
                continue
            token_start = first - left_extension
            token_end = last + right_extension
            if token_start < 0 or token_end >= len(tokens):
                continue
            spans.add((tokens[token_start][0], tokens[token_end][1]))
    for left_trim in range(_MAX_TOKEN_TRIM + 1):
        for right_trim in range(_MAX_TOKEN_TRIM + 1):
            if left_trim + right_trim == 0:
                continue
            token_start = first + left_trim
            token_end = last - right_trim
            if token_start <= token_end:
                spans.add((tokens[token_start][0], tokens[token_end][1]))
    return tuple(
        sorted(
            span
            for span in spans
            if _valid_variant_span(source_text, span)
        )
    )


def _clause_span(
    source_text: str,
    foundation: tuple[int, int],
) -> tuple[int, int]:
    start, end = foundation
    left_window_start = max(0, start - 160)
    left_matches = list(
        _CLAUSE_DELIMITER_RE.finditer(source_text, left_window_start, start)
    )
    clause_start = (
        left_matches[-1].end()
        if left_matches
        else left_window_start
    )
    right_window_end = min(len(source_text), end + 192)
    right_match = _CLAUSE_DELIMITER_RE.search(source_text, end, right_window_end)
    clause_end = right_match.start() if right_match is not None else right_window_end
    return clause_start, clause_end


def _add_variant(
    accumulators: dict[tuple[int, int, str], _VariantAccumulator],
    span: tuple[int, int] | list[int],
    entity_type: str,
    foundations: Sequence[Mapping[str, Any]],
    generators: set[str],
) -> None:
    start, end = int(span[0]), int(span[1])
    key = (start, end, entity_type)
    accumulator = accumulators.setdefault(
        key,
        _VariantAccumulator(
            sources=set(),
            source_evidence={},
            generators=set(),
            foundation_spans=set(),
            statuses=set(),
        ),
    )
    accumulator.generators.update(generators)
    for row in foundations:
        foundation_span = _position(row)
        accumulator.foundation_spans.add(foundation_span)
        accumulator.statuses.add(str(row.get("status", "source_only")))
        accumulator.all_source_agreement = accumulator.all_source_agreement or bool(
            row.get("all_source_agreement", False)
        )
        raw_sources = row.get("sources")
        assert isinstance(raw_sources, list)
        raw_evidence = row.get("source_evidence", {})
        if not isinstance(raw_evidence, Mapping):
            raise ValueError("Boundary foundation source_evidence must be an object")
        for raw_source in raw_sources:
            source = str(raw_source)
            accumulator.sources.add(source)
            evidence = raw_evidence.get(source, {})
            if not isinstance(evidence, Mapping):
                raise ValueError("Boundary source evidence must be an object")
            confidence = evidence.get("confidence")
            if confidence is not None and (
                not isinstance(confidence, int | float)
                or isinstance(confidence, bool)
            ):
                raise ValueError("Boundary source confidence must be numeric")
            labels = evidence.get("source_labels", [])
            if not isinstance(labels, list) or not all(
                isinstance(label, str) for label in labels
            ):
                raise ValueError("Boundary source labels must be strings")
            previous = accumulator.source_evidence.get(source)
            previous_confidence = previous[0] if previous is not None else None
            strongest_confidence = max(
                (
                    value
                    for value in (
                        previous_confidence,
                        float(confidence) if confidence is not None else None,
                    )
                    if value is not None
                ),
                default=None,
            )
            previous_labels = previous[1] if previous is not None else set()
            support_only = bool(evidence.get("support_only", False)) or (
                previous[2] if previous is not None else False
            )
            accumulator.source_evidence[source] = (
                strongest_confidence,
                {*previous_labels, *labels},
                support_only,
            )


def _finalize_variant(
    document_id: str,
    source_text: str,
    key: tuple[int, int, str],
    accumulator: _VariantAccumulator,
) -> Phase1BoundaryVariant:
    start, end, entity_type = key
    identity = f"{document_id}\0{entity_type}\0{start}:{end}"
    is_original = (start, end) in accumulator.foundation_spans
    status = (
        sorted(accumulator.statuses)[0]
        if is_original and len(accumulator.statuses) == 1
        else "boundary_variant"
    )
    return Phase1BoundaryVariant(
        document_id=document_id,
        variant_id=hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
        family_id="pending",
        text=source_text[start:end],
        entity_type=entity_type,
        position=(start, end),
        sources=tuple(sorted(accumulator.sources)),
        source_evidence=tuple(
            (
                source,
                evidence[0],
                tuple(sorted(evidence[1])),
                evidence[2],
            )
            for source, evidence in sorted(accumulator.source_evidence.items())
        ),
        generators=tuple(sorted(accumulator.generators)),
        foundation_spans=tuple(sorted(accumulator.foundation_spans)),
        status=status,
        all_source_agreement=accumulator.all_source_agreement,
    )


def _assign_boundary_families(
    document_id: str,
    variants: Sequence[Phase1BoundaryVariant],
) -> tuple[Phase1BoundaryVariant, ...]:
    output: list[Phase1BoundaryVariant] = []
    for entity_type in sorted(PHASE1_ALLOWED_TYPES):
        typed = sorted(
            (variant for variant in variants if variant.entity_type == entity_type),
            key=lambda variant: (
                variant.position[0],
                variant.position[1],
                variant.variant_id,
            ),
        )
        if not typed:
            continue
        foundation_components = _foundation_components(typed)
        variants_by_component: dict[
            tuple[tuple[int, int], ...],
            list[Phase1BoundaryVariant],
        ] = {component: [] for component in foundation_components}
        for variant in typed:
            component = min(
                foundation_components,
                key=lambda value: _foundation_component_rank(variant, value),
            )
            variants_by_component[component].append(variant)
        for component, family_variants in variants_by_component.items():
            output.extend(
                _finalize_family(
                    document_id,
                    entity_type,
                    component,
                    family_variants,
                )
            )
    return tuple(
        sorted(
            output,
            key=lambda variant: (
                variant.position[0],
                variant.position[1],
                -PHASE1_TYPE_PRIORITY[variant.entity_type],
                variant.variant_id,
            ),
        )
    )


def _foundation_components(
    variants: Sequence[Phase1BoundaryVariant],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Cluster original proposals only; generated windows must never bridge entities."""

    foundations = sorted(
        {
            foundation
            for variant in variants
            for foundation in variant.foundation_spans
        }
    )
    components: list[list[tuple[int, int]]] = []
    component_end = -1
    for foundation in foundations:
        if components and foundation[0] >= component_end:
            components.append([])
            component_end = -1
        elif not components:
            components.append([])
        components[-1].append(foundation)
        component_end = max(component_end, foundation[1])
    if not components:
        raise ValueError("Boundary variants require at least one foundation span")
    return tuple(tuple(component) for component in components)


def _foundation_component_rank(
    variant: Phase1BoundaryVariant,
    component: Sequence[tuple[int, int]],
) -> tuple[Any, ...]:
    overlap = max(_overlap_length(variant.position, span) for span in component)
    distance = min(
        abs(variant.position[0] - span[0])
        + abs(variant.position[1] - span[1])
        for span in component
    )
    owns_exact_foundation = int(variant.position in component)
    return (
        -owns_exact_foundation,
        -overlap,
        distance,
        component[0],
    )


def _finalize_family(
    document_id: str,
    entity_type: str,
    foundations: Sequence[tuple[int, int]],
    variants: Sequence[Phase1BoundaryVariant],
) -> tuple[Phase1BoundaryVariant, ...]:
    foundation_ids = "\n".join(
        f"{start}:{end}" for start, end in sorted(foundations)
    )
    family_id = hashlib.sha256(
        f"{document_id}\0{entity_type}\0{foundation_ids}".encode("utf-8")
    ).hexdigest()[:20]
    return tuple(replace(variant, family_id=family_id) for variant in variants)


def _valid_variant_span(source_text: str, span: tuple[int, int]) -> bool:
    start, end = span
    if start < 0 or end <= start or end > len(source_text):
        return False
    mention = source_text[start:end]
    return (
        bool(mention.strip())
        and len(mention) <= _MAX_VARIANT_CHARS
        and len(_WORD_RE.findall(mention)) <= _MAX_VARIANT_TOKENS
    )


def _trim_span(
    source_text: str,
    span: tuple[int, int],
) -> tuple[int, int] | None:
    start, end = span
    while start < end and source_text[start].isspace():
        start += 1
    while end > start and source_text[end - 1].isspace():
        end -= 1
    return (start, end) if _valid_variant_span(source_text, (start, end)) else None


def _position(row: Mapping[str, Any]) -> tuple[int, int]:
    raw = row.get("position")
    if (
        not isinstance(raw, list | tuple)
        or len(raw) != 2
        or not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in raw
        )
        or raw[0] < 0
        or raw[1] <= raw[0]
    ):
        raise ValueError("Boundary row has an invalid position")
    return int(raw[0]), int(raw[1])


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"Boundary variant {field} must be a list")
    return tuple(sorted({str(item) for item in value}))


def _span_tuple(value: Any, field: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"Boundary variant {field} must be a list")
    spans: set[tuple[int, int]] = set()
    for item in value:
        if (
            not isinstance(item, Sequence)
            or isinstance(item, str)
            or len(item) != 2
            or not all(isinstance(part, int) for part in item)
        ):
            raise ValueError(f"Boundary variant {field} contains an invalid span")
        spans.add((item[0], item[1]))
    return tuple(sorted(spans))


def _finite_probability(value: Any) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError("Boundary source confidence must be numeric")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Boundary source confidence must be within [0, 1]")
    return probability


def _overlap_length(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def _add_hash(features: dict[str, float], namespace: str, value: str) -> None:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % _HASH_BUCKETS
    features[f"hash:{namespace}:{bucket}"] = 1.0
