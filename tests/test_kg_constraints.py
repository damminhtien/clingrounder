from medical_kg_nlp.kg.constraints import entity_code_system_valid, relation_type_valid
from medical_kg_nlp.kg.reasoning import is_confirmed_patient_condition
from medical_kg_nlp.schema.annotation import EntityAnnotation, RelationAnnotation
from medical_kg_nlp.schema.types import AssertionStatus, CodeSystem, EntityType, RelationType


def test_drug_cannot_map_to_icd10() -> None:
    entity = EntityAnnotation(
        id="E1",
        span=(0, 9),
        text="metformin",
        normalized_text="metformin",
        type=EntityType.DRUG,
        assertion=AssertionStatus.PRESENT,
        code_system=CodeSystem.ICD10,
        code="E11",
    )
    assert not entity_code_system_valid(entity)


def test_treats_requires_drug_head() -> None:
    drug = EntityAnnotation("E1", (0, 9), "metformin", "metformin", EntityType.DRUG)
    disease = EntityAnnotation("E2", (20, 41), "đái tháo đường type 2", "đái tháo đường type 2", EntityType.DISEASE)
    relation = RelationAnnotation("R1", "E1", "E2", RelationType.TREATS, 0.9)
    assert relation_type_valid(relation, {"E1": drug, "E2": disease})


def test_only_present_assertion_is_confirmed_patient_condition() -> None:
    present = EntityAnnotation("E1", (0, 5), "dummy", "dummy", EntityType.DISEASE, assertion=AssertionStatus.PRESENT)
    historical = EntityAnnotation(
        "E2",
        (10, 15),
        "dummy",
        "dummy",
        EntityType.DISEASE,
        assertion=AssertionStatus.HISTORICAL,
    )
    possible = EntityAnnotation("E3", (20, 25), "dummy", "dummy", EntityType.DISEASE, assertion=AssertionStatus.POSSIBLE)
    family = EntityAnnotation("E4", (30, 35), "dummy", "dummy", EntityType.DISEASE, assertion=AssertionStatus.FAMILY)
    negated = EntityAnnotation("E5", (40, 45), "dummy", "dummy", EntityType.DISEASE, assertion=AssertionStatus.NEGATED)

    assert is_confirmed_patient_condition(present)
    assert not is_confirmed_patient_condition(historical)
    assert not is_confirmed_patient_condition(possible)
    assert not is_confirmed_patient_condition(family)
    assert not is_confirmed_patient_condition(negated)
