from __future__ import annotations
import re
from pathlib import Path

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.ner.dictionary_matcher import DictionaryMatch, DictionaryMatcher
from medical_kg_nlp.ner.lab_observation_extractor import LabObservationExtractor
from medical_kg_nlp.ontology.false_positive import DEFAULT_FALSE_POSITIVE_PATH, FalsePositiveRule, load_false_positive_rules
from medical_kg_nlp.schema.annotation import EntityAnnotation
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
    ) -> None:
        self.store = store
        self.aliases = store.aliases_for_ner()
        self.matcher = DictionaryMatcher(self.aliases)
        self.lab_observations = LabObservationExtractor()
        self._drug_alias_lowers = tuple(
            alias.lower()
            for alias, entry in self.aliases
            if entry.semantic_type == EntityType.DRUG and len(alias.strip()) >= 4
        )
        self.false_positive_rules: tuple[FalsePositiveRule, ...] = load_false_positive_rules(false_positive_path)

    def extract(self, text: str) -> list[EntityAnnotation]:
        spans: list[EntityAnnotation] = []
        occupied: list[tuple[int, int]] = []
        dictionary_matches = self.matcher.resolve_longest(
            match
            for match in self.matcher.find_candidates(text, require_boundaries=True, min_alias_chars=2)
            if not self._blocked_contextual_alias(match.alias, text, match.span)
        )
        for match in dictionary_matches:
            occupied.append(match.span)
            spans.append(
                EntityAnnotation(
                    id="",
                    span=match.span,
                    text=match.text,
                    normalized_text=match.normalized_text,
                    type=match.entry.semantic_type,
                    assertion=AssertionStatus.UNKNOWN,
                    code_system=CodeSystem.NONE,
                    confidence=0.78 if match.match_kind == "exact" else 0.76,
                )
            )
        self._extract_concatenated_drugs(text, occupied, spans)
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
        if any(rule.blocks(alias, text, span) for rule in self.false_positive_rules):
            return True
        normalized_alias = normalize_for_match(alias)
        if normalized_alias == "ho":
            right = text[span[1] : min(len(text), span[1] + 12)]
            right_token = right.lstrip()
            return bool(right_token and right_token[0].isalpha() and right_token[0].isupper())
        if normalized_alias.startswith("ung thư"):
            context = text[max(0, span[0] - 20) : min(len(text), span[1] + 20)].lower()
            return bool(re.search(r"kháng\s+nguyên\s+ung\s+thư\s+phôi", context, flags=re.UNICODE))
        if normalized_alias != "yếu":
            return False
        left = text[max(0, span[0] - 8) : span[0]].lower()
        right = text[span[1] : min(len(text), span[1] + 8)].lower()
        return bool(re.search(r"chủ\s*$", left, flags=re.UNICODE) or re.match(r"\s*tố(?!\w)", right, flags=re.UNICODE))

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
            spans.append(
                EntityAnnotation(
                    id="",
                    span=match.span,
                    text=match.text,
                    normalized_text=match.normalized_text,
                    type=match.entry.semantic_type,
                    assertion=AssertionStatus.UNKNOWN,
                    code_system=CodeSystem.NONE,
                    confidence=0.74,
                )
            )

    def _has_concatenated_drug_left_boundary(self, lowered: str, start: int) -> bool:
        left = lowered[max(0, start - 32) : start]
        return left.endswith(self._drug_alias_lowers) or left.endswith(_CONCATENATED_DRUG_LEFT_PREFIXES)

    def _has_concatenated_drug_right_boundary(self, lowered: str, end: int) -> bool:
        right = lowered[end : end + 32]
        return right.startswith(self._drug_alias_lowers) or right.startswith(_CONCATENATED_DRUG_RIGHT_SUFFIXES)

def _lab_result_span(match: re.Match[str]) -> tuple[int, int]:
    if "value" in match.re.groupindex:
        return match.span("value")
    return match.span()
