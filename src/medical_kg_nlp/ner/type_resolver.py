"""Context-aware semantic type resolution for dictionary proposals."""

from __future__ import annotations

import re
from dataclasses import dataclass

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
_SECTION_HEADER_RE = re.compile(
    r"^[ \t]*(?:[-*•]+[ \t]*)?(?:"
    r"(?P<symptom>"
    r"(?:các[ \t]+)?triệu[ \t]+chứng[ \t]+(?:hiện[ \t]+tại|khi[ \t]+nhập[ \t]+viện)|"
    r"đặc[ \t]+điểm[ \t]+triệu[ \t]+chứng|"
    r"(?:các[ \t]+)?triệu[ \t]+chứng[ \t]+(?:kèm[ \t]+theo|liên[ \t]+quan)"
    r")|"
    r"(?P<disease>"
    r"chẩn[ \t]+đoán|"
    r"(?:các[ \t]+)?bệnh[ \t]+lý[ \t]+(?:nội[ \t]+khoa[ \t]+)?m(?:ạ|ã)n[ \t]+tính|"
    r"tiền[ \t]+sử[ \t]+bệnh[ \t]+nội[ \t]+khoa"
    r")|"
    r"(?P<neutral>"
    r"tiền[ \t]+sử[ \t]+bệnh[ \t]+hiện[ \t]+tại|"
    r"lý[ \t]+do[ \t]+(?:nhập|vào)[ \t]+viện|"
    r"diễn[ \t]+biến[ \t]+bệnh|"
    r"(?:các[ \t]+)?sự[ \t]+kiện[ \t]+trước[ \t]+khi[ \t]+nhập[ \t]+viện|"
    r"tình[ \t]+trạng[ \t]+ngay[ \t]+trước[ \t]+khi[ \t]+nhập[ \t]+viện|"
    r"kết[ \t]+quả[ \t]+(?:xét[ \t]*nghiệm|chẩn[ \t]+đoán[ \t]+hình[ \t]+ảnh)|"
    r"cận[ \t]+lâm[ \t]+sàng|"
    r"đánh[ \t]+giá[ \t]+tại[ \t]+bệnh[ \t]+viện|"
    r"(?:các[ \t]+)?thủ[ \t]+thuật|"
    r"điều[ \t]+trị|thuốc|kế[ \t]+hoạch"
    r")"
    r")(?=[ \t]*:|[ \t]*$)",
    flags=re.IGNORECASE | re.MULTILINE | re.UNICODE,
)


@dataclass(frozen=True, slots=True)
class TypeResolution:
    """Resolved type and the stable rule that supplied the evidence."""

    entity_type: EntityType | None
    reason: str


@dataclass(frozen=True, slots=True)
class ContextualEntityTypeResolver:
    """Resolve cross-type aliases from local structure instead of fixed type authority."""

    context_radius: int = 64
    section_radius: int = 800

    def __post_init__(self) -> None:
        if self.context_radius < 16:
            raise ValueError("context_radius must be at least 16 characters")
        if self.section_radius < self.context_radius:
            raise ValueError("section_radius must be at least context_radius")

    def resolve(
        self,
        source_text: str,
        span: tuple[int, int],
        candidate_types: tuple[EntityType, ...],
        *,
        medication_indication_spans: tuple[tuple[int, int], ...] = (),
    ) -> TypeResolution:
        """Resolve one exact span while preserving ambiguous evidence when cues are absent."""

        left = source_text[max(0, span[0] - self.context_radius) : span[0]]
        right = source_text[span[1] : min(len(source_text), span[1] + self.context_radius)]
        candidate_set = set(candidate_types)
        in_medication_indication = any(
            indication_start <= span[0] and span[1] <= indication_end
            for indication_start, indication_end in medication_indication_spans
        )
        section_type = _nearest_section_type(
            source_text,
            span[0],
            radius=self.section_radius,
        )

        if len(candidate_types) == 1:
            # INVARIANT: context selects among observed type evidence; it never invents a
            # type that no extractor proposed. Reviewed code-free aliases can supply the
            # alternate evidence without weakening code-bearing terminology.
            return TypeResolution(candidate_types[0], "unique_dictionary_type")

        if candidate_set == {EntityType.DISEASE, EntityType.SYMPTOM}:
            if in_medication_indication:
                return TypeResolution(EntityType.SYMPTOM, "medication_indication")
            if section_type is EntityType.SYMPTOM:
                return TypeResolution(EntityType.SYMPTOM, "explicit_symptom_section")
            if section_type is EntityType.DISEASE:
                return TypeResolution(EntityType.DISEASE, "explicit_diagnosis_section")
            symptom_distance = _last_cue_distance(left, _SYMPTOM_LEFT_CUE_RE)
            disease_distance = _last_cue_distance(left, _DISEASE_LEFT_CUE_RE)
            if symptom_distance is not None and (
                disease_distance is None or symptom_distance < disease_distance
            ):
                return TypeResolution(EntityType.SYMPTOM, "explicit_symptom_context")
            if disease_distance is not None:
                return TypeResolution(EntityType.DISEASE, "explicit_diagnosis_context")
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


def _nearest_section_type(
    source_text: str,
    boundary: int,
    *,
    radius: int,
) -> EntityType | None:
    """Return the nearest explicit section type, with neutral headings terminating scope."""

    window_start = max(0, boundary - radius)
    matches = list(_SECTION_HEADER_RE.finditer(source_text, window_start, boundary))
    if not matches:
        return None
    nearest = matches[-1]
    if nearest.lastgroup == "symptom":
        return EntityType.SYMPTOM
    if nearest.lastgroup == "disease":
        return EntityType.DISEASE
    return None
