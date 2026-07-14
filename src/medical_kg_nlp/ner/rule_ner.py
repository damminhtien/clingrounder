from __future__ import annotations
import re
from pathlib import Path

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.ner.dictionary_matcher import DictionaryMatch, DictionaryMatcher
from medical_kg_nlp.ner.lab_observation_extractor import LabObservationExtractor
from medical_kg_nlp.ner.medication_attribute_extractor import MedicationAttributeExtractor
from medical_kg_nlp.ner.medication_mention_parser import MedicationMentionParser
from medical_kg_nlp.ner.medication_list_parser import MedicationListParser
from medical_kg_nlp.ontology.false_positive import (
    DEFAULT_FALSE_POSITIVE_PATH,
    FalsePositiveRule,
    load_false_positive_rules,
)
from medical_kg_nlp.schema.annotation import CandidateConcept, EntityAnnotation
from medical_kg_nlp.schema.types import AssertionStatus, CodeSystem, EntityType
from medical_kg_nlp.utils.text import normalize_for_match


_LAB_VALUE_RE = re.compile(
    r"(?<!\w)\d+(?:[\.,]\d+)?\s?(?:mmol/L|mg/dL|g/dL|ng/mL|mEq/L|IU/L|U/L|%)(?!\w)",
    flags=re.IGNORECASE,
)
_BP_RE = re.compile(r"(?<!\w)BP\s*(?P<value>\d{2,3}/\d{2,3})(?!\w)", flags=re.IGNORECASE)
_VITAL_VALUE_RE = re.compile(
    r"(?<!\w)"
    r"(?:"
    r"huyết\s+áp(?:\s+tâm\s+(?:thu|trương))?|"
    r"nhịp\s+thở|"
    r"nhịp\s+tim|"
    r"nhiệt\s+độ|"
    r"thân\s+nhiệt|"
    r"spo2|"
    r"độ\s+bão\s+h[oò]a\s+oxy|"
    r"bão\s+h[oò]a\s+oxy"
    r")"
    r"(?:\s+là)?\s*"
    r"(?P<value>\d{2,5}(?:/\d{2,3})?(?:[\.,]\d+)?(?:-\d{2,3})?(?:\s*%|\s*mmhg|\s*°?\s*c)?)"
    r"(?!\w)",
    flags=re.IGNORECASE | re.UNICODE,
)
_HBA1C_RE = re.compile(r"(?<!\w)HbA1c\s*\d+(?:\.\d+)?%(?!\w)", flags=re.IGNORECASE)
_CONCATENATED_DRUG_LEFT_PREFIXES = ("dùng", "uống", "tiêm", "truyền")
_CONCATENATED_DRUG_RIGHT_SUFFIXES = ("đã", "và", "trong", "kéo", "iv", "oral", "and")


class RuleBasedNER:
    def __init__(
        self,
        store: DictionaryStore,
        *,
        false_positive_path: str | Path | None = DEFAULT_FALSE_POSITIVE_PATH,
        emit_probabilities_by_source: dict[str, float] | None = None,
    ) -> None:
        self.store = store
        self.emit_probabilities_by_source = emit_probabilities_by_source or {}
        self.aliases = store.aliases_for_ner()
        self.matcher = DictionaryMatcher(self.aliases)
        self.lab_observations = LabObservationExtractor()
        self.medication_attributes = MedicationAttributeExtractor()
        self.medication_mentions = MedicationMentionParser()
        self.medication_lists = MedicationListParser()
        self._drug_alias_lowers = tuple(
            alias.lower()
            for alias, entry in self.aliases
            if entry.semantic_type == EntityType.DRUG and len(alias.strip()) >= 4
        )
        self.false_positive_rules: tuple[FalsePositiveRule, ...] = load_false_positive_rules(
            false_positive_path
        )

    def extract(self, text: str) -> list[EntityAnnotation]:
        spans: list[EntityAnnotation] = []
        occupied: list[tuple[int, int]] = []
        medication_list_items = self.medication_lists.items(text)
        indication_spans = tuple(
            item.indication_span
            for item in medication_list_items
            if item.indication_span is not None
        )
        raw_dictionary_matches = [
            match
            for match in self.matcher.find_candidates(
                text, require_boundaries=True, min_alias_chars=2
            )
            if not self._blocked_contextual_alias(match.alias, text, match.span)
        ]
        semantic_types_by_span: dict[tuple[int, int], set[EntityType]] = {}
        for match in raw_dictionary_matches:
            semantic_types_by_span.setdefault(match.span, set()).add(match.entry.semantic_type)
        selected_type_by_span = {
            span: self._disambiguated_semantic_type(
                text,
                span,
                entity_types,
                medication_indication_spans=indication_spans,
            )
            for span, entity_types in semantic_types_by_span.items()
        }
        dictionary_matches = self.matcher.resolve_longest(
            match
            for match in raw_dictionary_matches
            if match.entry.semantic_type == selected_type_by_span[match.span]
        )
        for match in dictionary_matches:
            occupied.append(match.span)
            spans.append(self._entity_from_dictionary_match(match))
        self._extract_concatenated_drugs(text, occupied, spans)
        spans.extend(self.medication_attributes.extract(text, spans, occupied=occupied))
        for entity in spans:
            if entity.type == EntityType.DRUG:
                entity.medication_mention = self.medication_mentions.parse(text, entity.span)
        spans = self.medication_lists.adjudicate(text, spans)
        for entity in self.lab_observations.extract(text, spans, occupied=occupied):
            occupied.append(entity.span)
            spans.append(entity)
        for regex in (_HBA1C_RE, _BP_RE, _VITAL_VALUE_RE, _LAB_VALUE_RE):
            for regex_match in regex.finditer(text):
                span = _lab_result_span(regex_match)
                if self._overlaps(span, occupied):
                    continue
                occupied.append(span)
                spans.append(
                    EntityAnnotation(
                        id="",
                        span=span,
                        text=text[span[0] : span[1]],
                        normalized_text=normalize_for_match(text[span[0] : span[1]]),
                        type=EntityType.LAB_RESULT,
                        assertion=AssertionStatus.PRESENT,
                        code_system=CodeSystem.NONE,
                        confidence=0.8,
                    )
                )
        spans.sort(key=lambda entity: (entity.span[0], entity.span[1]))
        for index, entity in enumerate(spans, start=1):
            entity.id = f"E{index}"
        return spans

    @staticmethod
    def _overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
        return any(span[0] < old_end and old_start < span[1] for old_start, old_end in occupied)

    def _blocked_contextual_alias(self, alias: str, text: str, span: tuple[int, int]) -> bool:
        return any(rule.blocks(alias, text, span) for rule in self.false_positive_rules)

    def _disambiguated_semantic_type(
        self,
        text: str,
        span: tuple[int, int],
        entity_types: set[EntityType],
        *,
        medication_indication_spans: tuple[tuple[int, int], ...] = (),
    ) -> EntityType | None:
        if len(entity_types) == 1:
            return next(iter(entity_types))
        if entity_types == {EntityType.DISEASE, EntityType.SYMPTOM}:
            # Dual-typed concepts used as medication indications follow the BTC symptom policy.
            # Diagnosis-only entries remain diseases, so unseen indications do not need a phrase
            # whitelist and retain their dictionary semantics.
            if any(
                indication_start <= span[0] and span[1] <= indication_end
                for indication_start, indication_end in medication_indication_spans
            ):
                return EntityType.SYMPTOM
            return EntityType.DISEASE
        left = text[max(0, span[0] - 32) : span[0]]
        right = text[span[1] : min(len(text), span[1] + 48)]
        if EntityType.DRUG in entity_types and re.search(
            r"(?<!\w)(?:dùng|uống|tiêm|truyền|thuốc|điều\s+trị\s+bằng)\s*$",
            left,
            flags=re.IGNORECASE | re.UNICODE,
        ):
            return EntityType.DRUG
        if EntityType.LAB_TEST in entity_types and (
            re.search(
                r"(?:là|:|=)?\s*(?:âm\s+tính|dương\s+tính|bình\s+thường|bất\s+thường|"
                r"tăng|giảm|cao|thấp|\d)",
                right,
                flags=re.IGNORECASE | re.UNICODE,
            )
            or re.search(
                r"(?<!\w)(?:xét\s+nghiệm|định\s+lượng)\s*$",
                left,
                flags=re.IGNORECASE | re.UNICODE,
            )
        ):
            return EntityType.LAB_TEST
        return None

    def _extract_concatenated_drugs(
        self,
        text: str,
        occupied: list[tuple[int, int]],
        spans: list[EntityAnnotation],
    ) -> None:
        lowered = text.lower()
        candidates: list[DictionaryMatch] = []
        for match in self.matcher.find_candidates(
            text,
            require_boundaries=False,
            entity_types={EntityType.DRUG},
            min_alias_chars=4,
        ):
            span = match.span
            if self._overlaps(span, occupied):
                continue
            left_concat = self._has_concatenated_drug_left_boundary(lowered, span[0])
            right_concat = self._has_concatenated_drug_right_boundary(lowered, span[1])
            left_boundary = span[0] == 0 or not text[span[0] - 1].isalnum() or left_concat
            right_boundary = span[1] == len(text) or not text[span[1]].isalnum() or right_concat
            if not (left_boundary and right_boundary and (left_concat or right_concat)):
                continue
            if self._blocked_contextual_alias(match.alias, text, span):
                continue
            candidates.append(match)
        for match in self.matcher.resolve_longest(candidates):
            if self._overlaps(match.span, occupied):
                continue
            occupied.append(match.span)
            spans.append(self._entity_from_dictionary_match(match, confidence=0.74))

    def _entity_from_dictionary_match(
        self,
        match: DictionaryMatch,
        *,
        confidence: float | None = None,
    ) -> EntityAnnotation:
        entry = self._unique_output_entry(match)
        score = 1.0 if match.match_kind == "exact" else 0.92
        source = f"dictionary_{match.match_kind}"
        candidate = (
            CandidateConcept(
                concept_id=entry.concept_id,
                code_system=entry.code_system,
                code=entry.code,
                name=entry.canonical_name,
                retrieval_score=score,
                emit_probability=self.emit_probabilities_by_source.get(
                    f"{entry.code_system.value}:{source}",
                    self.emit_probabilities_by_source.get(source, 0.0),
                ),
                source=source,
                evidence_sources=(source,),
                matched_alias=match.alias,
                qualified=True,
                qualification_reason="pinned_unique_dictionary_match",
            )
            if entry is not None
            else None
        )
        return EntityAnnotation(
            id="",
            span=match.span,
            text=match.text,
            normalized_text=match.normalized_text,
            type=match.entry.semantic_type,
            assertion=AssertionStatus.UNKNOWN,
            code_system=entry.code_system if entry is not None else CodeSystem.NONE,
            code=entry.code if entry is not None else None,
            confidence=(0.78 if match.match_kind == "exact" else 0.76)
            if confidence is None
            else confidence,
            candidates=[candidate] if candidate is not None else [],
        )

    def _unique_output_entry(self, match: DictionaryMatch) -> ConceptEntry | None:
        entries = (
            self.store.exact_lookup(match.alias)
            if match.match_kind == "exact"
            else self.store.toneless_lookup(match.alias)
        )
        compatible = [
            entry
            for entry in entries
            if entry.semantic_type == match.entry.semantic_type
            and entry.code is not None
            and entry.code_system != CodeSystem.NONE
        ]
        by_output = {(entry.code_system, entry.code): entry for entry in compatible}
        if len(by_output) != 1:
            return None
        return next(iter(by_output.values()))

    def _has_concatenated_drug_left_boundary(self, lowered: str, start: int) -> bool:
        left = lowered[max(0, start - 32) : start]
        return left.endswith(self._drug_alias_lowers) or left.endswith(
            _CONCATENATED_DRUG_LEFT_PREFIXES
        )

    def _has_concatenated_drug_right_boundary(self, lowered: str, end: int) -> bool:
        right = lowered[end : end + 32]
        return right.startswith(self._drug_alias_lowers) or right.startswith(
            _CONCATENATED_DRUG_RIGHT_SUFFIXES
        )


def _lab_result_span(match: re.Match[str]) -> tuple[int, int]:
    if "value" in match.re.groupindex:
        return match.span("value")
    return match.span()
