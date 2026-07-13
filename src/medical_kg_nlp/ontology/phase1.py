from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from medical_kg_nlp.schema.types import CodeSystem, EntityType, RelationType


PHASE1_REQUIRED_KEYS = frozenset({"text", "type", "assertions", "position"})
PHASE1_ALLOWED_KEYS = PHASE1_REQUIRED_KEYS | {"candidates"}
PHASE1_ALLOWED_ASSERTIONS = ("isNegated", "isFamily", "isHistorical")
PHASE1_TYPE_PRIORITY = {
    "THUỐC": 5,
    "CHẨN_ĐOÁN": 4,
    "TÊN_XÉT_NGHIỆM": 3,
    "KẾT_QUẢ_XÉT_NGHIỆM": 2,
    "TRIỆU_CHỨNG": 1,
}


@dataclass(frozen=True)
class Phase1EntityTypeRule:
    phase1_type: str
    internal_type: EntityType
    english: str
    candidate_system: CodeSystem | None
    allow_assertions: bool
    priority: int


@dataclass(frozen=True)
class Phase1SectionRule:
    label: str
    headings: tuple[str, ...]
    assertion_priors: tuple[str, ...] = ()
    type_prior: str | None = None


@dataclass(frozen=True)
class Phase1InternalRelationRule:
    relation: RelationType
    head_types: frozenset[str]
    tail_types: frozenset[str]
    use_for: tuple[str, ...]


@dataclass(frozen=True)
class OntologyRuleResult:
    rule_id: str
    passed: bool
    severity: str
    message: str


PHASE1_ENTITY_TYPE_RULES = (
    Phase1EntityTypeRule(
        phase1_type="TRIỆU_CHỨNG",
        internal_type=EntityType.SYMPTOM,
        english="Symptom",
        candidate_system=None,
        allow_assertions=True,
        priority=PHASE1_TYPE_PRIORITY["TRIỆU_CHỨNG"],
    ),
    Phase1EntityTypeRule(
        phase1_type="TÊN_XÉT_NGHIỆM",
        internal_type=EntityType.LAB_TEST,
        english="LabTestName",
        candidate_system=None,
        allow_assertions=False,
        priority=PHASE1_TYPE_PRIORITY["TÊN_XÉT_NGHIỆM"],
    ),
    Phase1EntityTypeRule(
        phase1_type="KẾT_QUẢ_XÉT_NGHIỆM",
        internal_type=EntityType.LAB_RESULT,
        english="LabResultValue",
        candidate_system=None,
        allow_assertions=False,
        priority=PHASE1_TYPE_PRIORITY["KẾT_QUẢ_XÉT_NGHIỆM"],
    ),
    Phase1EntityTypeRule(
        phase1_type="CHẨN_ĐOÁN",
        internal_type=EntityType.DISEASE,
        english="Diagnosis",
        candidate_system=CodeSystem.ICD10,
        allow_assertions=True,
        priority=PHASE1_TYPE_PRIORITY["CHẨN_ĐOÁN"],
    ),
    Phase1EntityTypeRule(
        phase1_type="THUỐC",
        internal_type=EntityType.DRUG,
        english="Medication",
        candidate_system=CodeSystem.RXNORM,
        allow_assertions=True,
        priority=PHASE1_TYPE_PRIORITY["THUỐC"],
    ),
)

PHASE1_TYPE_BY_ENTITY_TYPE = {
    rule.internal_type: rule.phase1_type for rule in PHASE1_ENTITY_TYPE_RULES
}
PHASE1_RULE_BY_TYPE = {rule.phase1_type: rule for rule in PHASE1_ENTITY_TYPE_RULES}
PHASE1_ALLOWED_TYPES = frozenset(PHASE1_RULE_BY_TYPE)
PHASE1_ASSERTABLE_TYPES = frozenset(rule.phase1_type for rule in PHASE1_ENTITY_TYPE_RULES if rule.allow_assertions)
PHASE1_CODABLE_TYPES = frozenset(
    rule.phase1_type for rule in PHASE1_ENTITY_TYPE_RULES if rule.candidate_system is not None
)
PHASE1_CODE_SYSTEM_BY_TYPE = {
    rule.phase1_type: rule.candidate_system
    for rule in PHASE1_ENTITY_TYPE_RULES
    if rule.candidate_system is not None
}

PHASE1_SECTION_RULES = (
    Phase1SectionRule(
        label="HISTORY",
        headings=("tiền sử bệnh", "tiền sử bệnh nội khoa", "tiền sử", "bệnh sử"),
        assertion_priors=("isHistorical",),
    ),
    Phase1SectionRule(
        label="MEDICATION_HISTORY",
        headings=(
            "danh sách thuốc trước nhập viện",
            "danh sách thuốc trước khi nhập viện",
            "thuốc trước khi nhập viện",
            "thuốc đang dùng trước khi nhập viện",
            "tiền sử dùng thuốc",
        ),
        assertion_priors=("isHistorical",),
        type_prior="THUỐC",
    ),
    Phase1SectionRule(
        label="PRESENT_ILLNESS",
        headings=("bệnh sử hiện tại", "tiền sử bệnh hiện tại", "triệu chứng hiện tại", "lý do nhập viện"),
    ),
    Phase1SectionRule(
        label="FAMILY_HISTORY",
        headings=("tiền sử gia đình", "bệnh sử gia đình", "gia đình"),
        assertion_priors=("isFamily", "isHistorical"),
    ),
    Phase1SectionRule(
        label="LAB",
        headings=("kết quả xét nghiệm", "xét nghiệm", "cận lâm sàng", "sinh hóa máu", "huyết học"),
        type_prior="TÊN_XÉT_NGHIỆM",
    ),
    Phase1SectionRule(
        label="ASSESSMENT",
        headings=("đánh giá tại bệnh viện", "chẩn đoán", "kết luận", "đánh giá"),
        type_prior="CHẨN_ĐOÁN",
    ),
)

PHASE1_INTERNAL_RELATION_RULES = (
    Phase1InternalRelationRule(
        relation=RelationType.TREATS,
        head_types=frozenset({"THUỐC"}),
        tail_types=frozenset({"CHẨN_ĐOÁN", "TRIỆU_CHỨNG"}),
        use_for=("candidate_reranking", "consistency_check"),
    ),
    Phase1InternalRelationRule(
        relation=RelationType.HAS_SYMPTOM,
        head_types=frozenset({"CHẨN_ĐOÁN"}),
        tail_types=frozenset({"TRIỆU_CHỨNG"}),
        use_for=("diagnosis_reranking", "symptom_type_validation"),
    ),
    Phase1InternalRelationRule(
        relation=RelationType.HAS_VALUE,
        head_types=frozenset({"TÊN_XÉT_NGHIỆM"}),
        tail_types=frozenset({"KẾT_QUẢ_XÉT_NGHIỆM"}),
        use_for=("span_grouping", "lab_parser_validation"),
    ),
    Phase1InternalRelationRule(
        relation=RelationType.SUGGESTS,
        head_types=frozenset({"TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}),
        tail_types=frozenset({"CHẨN_ĐOÁN"}),
        use_for=("diagnosis_reranking",),
    ),
    Phase1InternalRelationRule(
        relation=RelationType.HAS_DOSE,
        head_types=frozenset({"THUỐC"}),
        tail_types=frozenset({"KẾT_QUẢ_XÉT_NGHIỆM"}),
        use_for=("medication_span_expansion", "rxnorm_reranking"),
    ),
)


def expected_code_system(phase1_type: str) -> CodeSystem | None:
    return PHASE1_CODE_SYSTEM_BY_TYPE.get(phase1_type)


def candidates_allowed(phase1_type: str) -> bool:
    return phase1_type in PHASE1_CODABLE_TYPES


def assertions_allowed(phase1_type: str) -> bool:
    return phase1_type in PHASE1_ASSERTABLE_TYPES


def assertion_sort_key(assertion: str) -> int:
    try:
        return PHASE1_ALLOWED_ASSERTIONS.index(assertion)
    except ValueError:
        return len(PHASE1_ALLOWED_ASSERTIONS)


def section_rule_for_heading(heading: str) -> Phase1SectionRule | None:
    normalized = heading.lower().strip()
    if not normalized:
        return None
    candidates: list[tuple[int, Phase1SectionRule]] = []
    for rule in PHASE1_SECTION_RULES:
        for candidate in rule.headings:
            if candidate in normalized:
                candidates.append((len(candidate), rule))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def internal_relation_allowed(relation: RelationType, head_type: str, tail_type: str) -> bool:
    for rule in PHASE1_INTERNAL_RELATION_RULES:
        if rule.relation == relation and head_type in rule.head_types and tail_type in rule.tail_types:
            return True
    return False


def resolve_overlap(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any] | None:
    left_position = left.get("position")
    right_position = right.get("position")
    if not (_is_position(left_position) and _is_position(right_position)):
        return None
    left_start, left_end = cast(list[int], left_position)
    right_start, right_end = cast(list[int], right_position)
    overlap = max(0, min(left_end, right_end) - max(left_start, right_start))
    if overlap == 0:
        return None
    if left.get("type") == right.get("type"):
        left_length = left_end - left_start
        right_length = right_end - right_start
        return left if left_length >= right_length else right
    left_priority = PHASE1_TYPE_PRIORITY.get(str(left.get("type")), 0)
    right_priority = PHASE1_TYPE_PRIORITY.get(str(right.get("type")), 0)
    return left if left_priority >= right_priority else right


def _is_position(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(isinstance(item, int) for item in value)
