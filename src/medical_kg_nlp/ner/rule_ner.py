from __future__ import annotations
import re
from pathlib import Path

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.ontology.false_positive import DEFAULT_FALSE_POSITIVE_PATH, FalsePositiveRule, load_false_positive_rules
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.types import AssertionStatus, CodeSystem, EntityType
from medical_kg_nlp.utils.text import normalize_for_match


_LAB_VALUE_RE = re.compile(
    r"(?<!\w)\d+(?:[\.,]\d+)?\s?(?:mmol/L|mg/dL|g/dL|ng/mL|mEq/L|IU/L|U/L|%)(?!\w)",
    flags=re.IGNORECASE,
)
_BP_RE = re.compile(r"(?<!\w)BP\s*\d{2,3}/\d{2,3}(?!\w)", flags=re.IGNORECASE)
_HBA1C_RE = re.compile(r"(?<!\w)HbA1c\s*\d+(?:\.\d+)?%(?!\w)", flags=re.IGNORECASE)
_UPPERCASE_LETTERS = "A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ"
_RIGHT_ALIAS_BOUNDARY = rf"(?=$|[^\w]|[{_UPPERCASE_LETTERS}])"


class RuleBasedNER:
    def __init__(
        self,
        store: DictionaryStore,
        *,
        false_positive_path: str | Path | None = DEFAULT_FALSE_POSITIVE_PATH,
    ) -> None:
        self.store = store
        self.aliases = store.aliases_for_ner()
        self.false_positive_rules: tuple[FalsePositiveRule, ...] = load_false_positive_rules(false_positive_path)

    def extract(self, text: str) -> list[EntityAnnotation]:
        spans: list[EntityAnnotation] = []
        occupied: list[tuple[int, int]] = []
        for alias, entry in self.aliases:
            if len(alias.strip()) < 2:
                continue
            pattern = re.compile(
                rf"(?<!\w)(?i:{re.escape(alias)}){_right_boundary_for_alias(alias)}",
                flags=re.UNICODE,
            )
            for match in pattern.finditer(text):
                span = match.span()
                if self._blocked_contextual_alias(alias, text, span):
                    continue
                if self._overlaps(span, occupied):
                    continue
                occupied.append(span)
                spans.append(
                    EntityAnnotation(
                        id="",
                        span=span,
                        text=text[span[0] : span[1]],
                        normalized_text=normalize_for_match(text[span[0] : span[1]]),
                        type=entry.semantic_type,
                        assertion=AssertionStatus.UNKNOWN,
                        code_system=CodeSystem.NONE,
                        confidence=0.78,
                    )
                )
        for regex in (_HBA1C_RE, _BP_RE, _LAB_VALUE_RE):
            for match in regex.finditer(text):
                span = match.span()
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
        if normalized_alias == "ung thư":
            context = text[max(0, span[0] - 20) : min(len(text), span[1] + 20)].lower()
            return bool(re.search(r"kháng\s+nguyên\s+ung\s+thư\s+phôi", context, flags=re.UNICODE))
        if normalized_alias != "yếu":
            return False
        left = text[max(0, span[0] - 8) : span[0]].lower()
        right = text[span[1] : min(len(text), span[1] + 8)].lower()
        return bool(re.search(r"chủ\s*$", left, flags=re.UNICODE) or re.match(r"\s*tố(?!\w)", right, flags=re.UNICODE))


def _right_boundary_for_alias(alias: str) -> str:
    compact = re.sub(r"[\W_]+", "", alias, flags=re.UNICODE)
    if compact and compact.upper() == compact:
        return r"(?!\w)"
    return _RIGHT_ALIAS_BOUNDARY
