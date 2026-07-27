"""Context-aware semantic type resolution for dictionary proposals."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from medical_kg_nlp.schema.types import EntityType

__all__ = [
    "ContextualEntityTypeResolver",
    "TypeResolution",
]


_SYMPTOM_LEFT_CUE_RE = re.compile(
    r"(?<!\w)(?:triệu\s+chứng|than\s+phiền|cảm\s+thấy|xuất\s+hiện|biểu\s+hiện|"
    r"ghi\s+nhận|hiện\s+có)\s*(?::|-)?\s*$",
    flags=re.IGNORECASE | re.UNICODE,
)
_DISEASE_LEFT_CUE_RE = re.compile(
    r"(?<!\w)(?:chẩn\s+đoán|bệnh\s+lý|bệnh\s+nền|tiền\s+sử|mắc|được\s+chẩn\s+đoán)"
    r"\s*(?::|-)?\s*$",
    flags=re.IGNORECASE | re.UNICODE,
)
_DRUG_LEFT_CUE_RE = re.compile(
    r"(?<!\w)(?:dùng|uống|tiêm|truyền|thuốc|điều\s+trị\s+bằng)\s*$",
    flags=re.IGNORECASE | re.UNICODE,
)
_LAB_LEFT_CUE_RE = re.compile(
    r"(?<!\w)(?:xét\s+nghiệm|định\s+lượng|hoạt\s+độ|nồng\s+độ)\s*$",
    flags=re.IGNORECASE | re.UNICODE,
)
_LAB_RIGHT_EVIDENCE_RE = re.compile(
    r"\s*(?:\([^()\n\r]{1,40}\)\s*)?"
    r"(?:(?:là|:|=|→|->|cải\s+thiện\s+thành)\s*)?"
    r"(?:âm\s+tính|dương\s+tính|bình\s+thường|bất\s+thường|"
    r"tăng|giảm|cao|thấp|[<>]?\s*\d)",
    flags=re.IGNORECASE | re.UNICODE,
)


@dataclass(frozen=True, slots=True)
class TypeResolution:
    """Resolved type and the stable rule that supplied the evidence."""

    entity_type: EntityType | None
    reason: str


@dataclass(frozen=True, slots=True)
class ContextualEntityTypeResolver:
    """Resolve cross-type aliases from local structure instead of fixed type authority."""

    disease_symptom_fallback: Literal["disease", "abstain"] = "disease"
    context_radius: int = 64

    def __post_init__(self) -> None:
        if self.context_radius < 16:
            raise ValueError("context_radius must be at least 16 characters")

    def resolve(
        self,
        source_text: str,
        span: tuple[int, int],
        candidate_types: tuple[EntityType, ...],
        *,
        medication_indication_spans: tuple[tuple[int, int], ...] = (),
    ) -> TypeResolution:
        """Resolve one exact span while preserving ambiguous evidence when cues are absent."""

        if len(candidate_types) == 1:
            return TypeResolution(candidate_types[0], "unique_dictionary_type")

        left = source_text[max(0, span[0] - self.context_radius) : span[0]]
        right = source_text[span[1] : min(len(source_text), span[1] + self.context_radius)]
        candidate_set = set(candidate_types)

        if candidate_set == {EntityType.DISEASE, EntityType.SYMPTOM}:
            if any(
                indication_start <= span[0] and span[1] <= indication_end
                for indication_start, indication_end in medication_indication_spans
            ):
                return TypeResolution(EntityType.SYMPTOM, "medication_indication")
            symptom_distance = _last_cue_distance(left, _SYMPTOM_LEFT_CUE_RE)
            disease_distance = _last_cue_distance(left, _DISEASE_LEFT_CUE_RE)
            if symptom_distance is not None and (
                disease_distance is None or symptom_distance < disease_distance
            ):
                return TypeResolution(EntityType.SYMPTOM, "explicit_symptom_context")
            if disease_distance is not None:
                return TypeResolution(EntityType.DISEASE, "explicit_diagnosis_context")
            if self.disease_symptom_fallback == "disease":
                return TypeResolution(EntityType.DISEASE, "legacy_disease_fallback")
            return TypeResolution(None, "disease_symptom_context_missing")

        if EntityType.DRUG in candidate_set and _DRUG_LEFT_CUE_RE.search(left):
            return TypeResolution(EntityType.DRUG, "drug_administration_context")
        if EntityType.LAB_TEST in candidate_set and (
            _LAB_LEFT_CUE_RE.search(left) or _LAB_RIGHT_EVIDENCE_RE.match(right)
        ):
            return TypeResolution(EntityType.LAB_TEST, "laboratory_context")
        return TypeResolution(None, "cross_type_context_missing")


def _last_cue_distance(left_context: str, pattern: re.Pattern[str]) -> int | None:
    match = pattern.search(left_context)
    if match is None:
        return None
    return len(left_context) - match.end()
