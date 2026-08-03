from medical_kg_nlp.benchmarks.phase1.ontology import (
    PHASE1_ALLOWED_ASSERTIONS,
    PHASE1_ALLOWED_TYPES,
    PHASE1_REQUIRED_KEYS,
    assertions_allowed,
    candidates_allowed,
    expected_code_system,
    internal_relation_allowed,
    resolve_overlap,
    section_rule_for_heading,
)
from medical_kg_nlp.schema.types import CodeSystem, RelationType


def test_phase1_operational_ontology_code_system_constraints() -> None:
    assert PHASE1_ALLOWED_TYPES == {
        "TRIỆU_CHỨNG",
        "TÊN_XÉT_NGHIỆM",
        "KẾT_QUẢ_XÉT_NGHIỆM",
        "CHẨN_ĐOÁN",
        "THUỐC",
    }
    assert PHASE1_REQUIRED_KEYS == {"text", "type", "assertions", "position"}
    assert PHASE1_ALLOWED_ASSERTIONS == ("isNegated", "isFamily", "isHistorical")

    assert expected_code_system("CHẨN_ĐOÁN") == CodeSystem.ICD10
    assert expected_code_system("THUỐC") == CodeSystem.RXNORM
    assert expected_code_system("TRIỆU_CHỨNG") is None
    assert candidates_allowed("CHẨN_ĐOÁN")
    assert candidates_allowed("THUỐC")
    assert not candidates_allowed("TÊN_XÉT_NGHIỆM")


def test_phase1_operational_ontology_assertion_allowed_types() -> None:
    assert assertions_allowed("TRIỆU_CHỨNG")
    assert assertions_allowed("CHẨN_ĐOÁN")
    assert assertions_allowed("THUỐC")
    assert not assertions_allowed("TÊN_XÉT_NGHIỆM")
    assert not assertions_allowed("KẾT_QUẢ_XÉT_NGHIỆM")


def test_phase1_operational_ontology_section_priors() -> None:
    medication = section_rule_for_heading("Thuốc trước khi nhập viện lần này")
    family = section_rule_for_heading("Tiền sử gia đình")
    present = section_rule_for_heading("Tiền sử bệnh hiện tại")

    assert medication is not None
    assert medication.label == "MEDICATION_HISTORY"
    assert medication.type_prior == "THUỐC"
    assert medication.assertion_priors == ("isHistorical",)
    assert family is not None
    assert family.assertion_priors == ("isFamily", "isHistorical")
    assert present is not None
    assert present.label == "PRESENT_ILLNESS"


def test_phase1_operational_ontology_internal_relation_constraints() -> None:
    assert internal_relation_allowed(RelationType.TREATS, "THUỐC", "CHẨN_ĐOÁN")
    assert internal_relation_allowed(RelationType.TREATS, "THUỐC", "TRIỆU_CHỨNG")
    assert internal_relation_allowed(RelationType.HAS_VALUE, "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM")
    assert not internal_relation_allowed(RelationType.TREATS, "KẾT_QUẢ_XÉT_NGHIỆM", "CHẨN_ĐOÁN")
    assert not internal_relation_allowed(RelationType.HAS_VALUE, "THUỐC", "KẾT_QUẢ_XÉT_NGHIỆM")


def test_phase1_operational_ontology_overlap_resolution() -> None:
    drug = {"text": "metformin", "type": "THUỐC", "position": [10, 19]}
    diagnosis = {"text": "metformin", "type": "CHẨN_ĐOÁN", "position": [10, 19]}
    short = {"text": "đái tháo đường", "type": "CHẨN_ĐOÁN", "position": [30, 44]}
    long = {"text": "đái tháo đường type 2", "type": "CHẨN_ĐOÁN", "position": [30, 51]}
    disjoint = {"text": "ho", "type": "TRIỆU_CHỨNG", "position": [80, 82]}

    assert resolve_overlap(drug, diagnosis) == drug
    assert resolve_overlap(short, long) == long
    assert resolve_overlap(drug, disjoint) is None
