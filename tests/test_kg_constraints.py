from medical_kg_nlp.kg.constraints import entity_code_system_valid, relation_type_valid
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

