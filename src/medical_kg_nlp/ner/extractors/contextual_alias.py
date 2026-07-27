"""Context-gated exact and toneless alias proposals."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.ner.contracts import RuleNerContext
from medical_kg_nlp.ner.dictionary_matcher import DictionaryMatcher
from medical_kg_nlp.ner.document_structure import (
    DocumentGenre,
    DocumentStructureAnalyzer,
    SectionKind,
)
from medical_kg_nlp.ner.proposal import EntityProposal
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.utils.io import read_yaml

__all__ = [
    "ContextGate",
    "ContextualAliasProposalExtractor",
    "ContextualAliasRule",
    "load_contextual_alias_rules",
]


class ContextGate(StrEnum):
    """Auditable structural evidence accepted by a contextual alias rule."""

    LAB_SECTION = "lab_section"
    LAB_VALUE_NEIGHBOR = "lab_value_neighbor"
    MEDICATION_INDICATION = "medication_indication"
    QUESTION_ANSWER = "question_answer"
    STANDALONE_LIST_ITEM = "standalone_list_item"
    SYMPTOM_PREDICATE = "symptom_predicate"
    SYMPTOM_SECTION = "symptom_section"
    VITAL_SIGNS_LINE = "vital_signs_line"


_MatchMode = Literal["exact", "toneless"]


@dataclass(frozen=True, slots=True)
class ContextualAliasRule:
    """One reviewed alias that requires at least one structural context gate."""

    rule_id: str
    alias: str
    entity_type: EntityType
    required_any: tuple[ContextGate, ...]
    match_modes: tuple[_MatchMode, ...] = ("exact",)
    score: float = 0.72
    provenance: str = ""

    def __post_init__(self) -> None:
        if not self.rule_id.strip() or not self.alias.strip():
            raise ValueError("Contextual alias rule_id and alias must be non-empty")
        if "\n" in self.alias or "\r" in self.alias:
            raise ValueError("Contextual aliases cannot cross source lines")
        if not self.required_any or tuple(sorted(set(self.required_any))) != self.required_any:
            raise ValueError("required_any must be non-empty, unique, and sorted")
        if (
            not self.match_modes
            or tuple(sorted(set(self.match_modes))) != self.match_modes
            or any(mode not in {"exact", "toneless"} for mode in self.match_modes)
        ):
            raise ValueError("match_modes must be non-empty, unique, and sorted")
        if not 0.0 < self.score < 1.0:
            raise ValueError("Contextual alias score must be between 0 and 1")


class ContextualAliasProposalExtractor:
    """Emit short or ambiguous aliases only when reviewed context evidence exists.

    SCALING: aliases are compiled into Aho-Corasick automatons. Short toneless matching is enabled
    only in this gated source; the primary dictionary keeps its conservative length threshold.
    """

    def __init__(self, rules: Sequence[ContextualAliasRule]) -> None:
        ordered = tuple(sorted(rules, key=lambda rule: rule.rule_id))
        rule_ids = [rule.rule_id for rule in ordered]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("Contextual alias rules contain duplicate rule IDs")
        self.rules = ordered
        self._rules_by_concept_id = {
            _concept_id(rule.rule_id): rule for rule in ordered
        }
        entries = tuple(
            (
                rule.alias,
                ConceptEntry(
                    concept_id=_concept_id(rule.rule_id),
                    code=None,
                    code_system=CodeSystem.NONE,
                    canonical_name=rule.alias,
                    semantic_type=rule.entity_type,
                    source=f"contextual_alias:{rule.provenance or 'reviewed'}",
                ),
            )
            for rule in ordered
        )
        self.matcher = DictionaryMatcher(
            entries,
            min_toneless_alias_chars=2,
            allow_compact_toneless_aliases=True,
        )
        self.structure_analyzer = DocumentStructureAnalyzer()

    def propose(
        self,
        source_text: str,
        context: RuleNerContext,
    ) -> tuple[EntityProposal, ...]:
        if not self.rules:
            return ()
        structure = context.structure or self.structure_analyzer.analyze(source_text)
        proposals: set[EntityProposal] = set()
        for match in self.matcher.find_candidates(source_text, require_boundaries=True):
            rule = self._rules_by_concept_id[match.entry.concept_id]
            if match.match_kind not in rule.match_modes:
                continue
            evidence = _context_evidence(
                source_text,
                match.span,
                rule,
                context,
                structure,
            )
            accepted = tuple(sorted(set(rule.required_any) & set(evidence)))
            if not accepted:
                continue
            proposal = EntityProposal(
                span=match.span,
                candidate_types=(rule.entity_type,),
                source="contextual_alias",
                score=rule.score,
                evidence_ids=tuple(
                    sorted(
                        {
                            f"contextual_alias:{rule.rule_id}",
                            *(f"context_gate:{gate.value}" for gate in accepted),
                        }
                    )
                ),
                features=(
                    ("context_gates", ",".join(gate.value for gate in accepted)),
                    ("match_kind", match.match_kind),
                    ("rule_id", rule.rule_id),
                ),
            )
            proposal.validate_offsets(source_text)
            proposals.add(proposal)
        return tuple(
            sorted(
                proposals,
                key=lambda item: (
                    item.span[0],
                    item.span[1],
                    item.candidate_types[0].value,
                ),
            )
        )


_SCHEMA_VERSION = "contextual-alias-rules.v1"
_FORBIDDEN_RULE_FIELDS = frozenset(
    {
        "document_id",
        "document_ids",
        "end",
        "offset",
        "offsets",
        "position",
        "positions",
        "span",
        "spans",
        "start",
    }
)
_SYMPTOM_PREDICATE_RE = re.compile(
    r"(?<!\w)(?:bị|có|than|cảm\s+thấy|xuất\s+hiện|biểu\s+hiện|"
    r"ghi\s+nhận|hiện\s+có|phủ\s+nhận|không\s+có)\s*(?::|-)?\s*$",
    flags=re.IGNORECASE | re.UNICODE,
)
_LAB_VALUE_RE = re.compile(
    r"(?:"
    r"(?<!\w)[<>]?\s*\d+(?:[.,]\d+)*(?:\s*%|\s*[a-zµμ/]+)?|"
    r"(?<!\w)(?:âm\s+tính|dương\s+tính|bình\s+thường|bất\s+thường|"
    r"tăng|giảm|cao|thấp)(?!\w)"
    r")",
    flags=re.IGNORECASE | re.UNICODE,
)
_VITAL_CONTEXT_RE = re.compile(
    r"(?<!\w)(?:dấu\s+hiệu\s+sinh\s+tồn|sinh\s+hiệu|vital\s+signs?)(?!\w)",
    flags=re.IGNORECASE | re.UNICODE,
)
_LIST_REMAINDER_RE = re.compile(
    r"^\s*(?:$|[:;,.)]|(?:bắt\s+đầu|vẫn|tiếp\s+tục|ngày\s+càng|"
    r"kéo\s+dài|tái\s+phát|xảy\s+ra|xuất\s+hiện)\b)",
    flags=re.IGNORECASE | re.UNICODE,
)


def load_contextual_alias_rules(path: str | Path) -> tuple[ContextualAliasRule, ...]:
    """Load a versioned rule artifact and reject document-specific shortcuts."""

    raw = read_yaml(path)
    if not isinstance(raw, Mapping) or raw.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("Unsupported contextual alias rule schema")
    _reject_document_specific_fields(raw)
    raw_rules = raw.get("rules")
    if not isinstance(raw_rules, list):
        raise ValueError("Contextual alias artifact requires a rules array")
    rules = tuple(_rule_from_mapping(item) for item in raw_rules)
    if not rules:
        raise ValueError("Contextual alias artifact must contain at least one rule")
    return rules


def _rule_from_mapping(raw: object) -> ContextualAliasRule:
    if not isinstance(raw, Mapping):
        raise ValueError("Contextual alias rule must be an object")
    raw_required = _string_sequence(raw, "required_any")
    raw_modes = _string_sequence(raw, "match_modes")
    return ContextualAliasRule(
        rule_id=_required_string(raw, "rule_id"),
        alias=_required_string(raw, "alias"),
        entity_type=EntityType(_required_string(raw, "entity_type")),
        required_any=tuple(sorted(ContextGate(value) for value in raw_required)),
        match_modes=tuple(sorted(raw_modes)),  # type: ignore[arg-type]
        score=float(raw.get("score", 0.72)),
        provenance=str(raw.get("provenance", "")),
    )


def _context_evidence(
    source_text: str,
    span: tuple[int, int],
    rule: ContextualAliasRule,
    context: RuleNerContext,
    structure: Any,
) -> tuple[ContextGate, ...]:
    evidence: list[ContextGate] = []
    section = structure.section_at(span[0])
    line = structure.line_at(span[0])
    left = source_text[max(0, span[0] - 80) : span[0]]
    if (
        section is not None
        and section.kind is SectionKind.SYMPTOM
        and (
            structure.starts_list_item(span[0])
            or section.heading_span[0] <= span[0] <= section.heading_span[1]
        )
    ):
        # A section heading may be followed by unrelated narrative without another heading.
        # Require either a structured item or an inline mention on the heading line.
        evidence.append(ContextGate.SYMPTOM_SECTION)
    if section is not None and section.kind is SectionKind.LABORATORY:
        evidence.append(ContextGate.LAB_SECTION)
    if structure.genre is DocumentGenre.QUESTION_ANSWER:
        evidence.append(ContextGate.QUESTION_ANSWER)
    if _SYMPTOM_PREDICATE_RE.search(_same_clause(left)):
        evidence.append(ContextGate.SYMPTOM_PREDICATE)
    if any(
        item.indication_span is not None
        and item.indication_span[0] <= span[0]
        and span[1] <= item.indication_span[1]
        for item in context.medication_items
    ):
        evidence.append(ContextGate.MEDICATION_INDICATION)
    if structure.starts_list_item(span[0]) and line is not None:
        if _LIST_REMAINDER_RE.match(source_text[span[1] : line.span[1]]):
            evidence.append(ContextGate.STANDALONE_LIST_ITEM)
    if line is not None:
        line_text = source_text[line.span[0] : line.span[1]]
        local_start = span[0] - line.span[0]
        local_end = span[1] - line.span[0]
        neighbor = line_text[max(0, local_start - 48) : min(len(line_text), local_end + 48)]
        mention_start = max(0, local_start - max(0, local_start - 48))
        mention_end = mention_start + (span[1] - span[0])
        without_mention = neighbor[:mention_start] + " " + neighbor[mention_end:]
        if _LAB_VALUE_RE.search(without_mention):
            evidence.append(ContextGate.LAB_VALUE_NEIGHBOR)
        if _VITAL_CONTEXT_RE.search(line_text):
            evidence.append(ContextGate.VITAL_SIGNS_LINE)
    return tuple(sorted(set(evidence)))


def _same_clause(left: str) -> str:
    boundary = max(left.rfind(token) for token in ("\n", "\r", ".", ";", ","))
    return left[boundary + 1 :]


def _concept_id(rule_id: str) -> str:
    return f"CONTEXT_ALIAS:{rule_id}"


def _reject_document_specific_fields(raw: object) -> None:
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if str(key).casefold() in _FORBIDDEN_RULE_FIELDS:
                raise ValueError(
                    f"Contextual alias artifact cannot contain document-specific field {key!r}"
                )
            _reject_document_specific_fields(value)
    elif isinstance(raw, Sequence) and not isinstance(raw, str | bytes):
        for value in raw:
            _reject_document_specific_fields(value)


def _required_string(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Contextual alias rule requires non-empty {key}")
    return value.strip()


def _string_sequence(raw: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Contextual alias rule requires non-empty {key} array")
    output = tuple(str(item) for item in value)
    if any(not item.strip() for item in output):
        raise ValueError(f"Contextual alias rule {key} cannot contain empty values")
    return output
