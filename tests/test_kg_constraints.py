from clingrounder.kg.constraints import (
    code_system_valid_for_entity_type,
    entity_code_system_valid,
    relation_type_valid,
)
from clingrounder.retrieval.constraints import allowed_code_systems
from clingrounder.kg.reasoning import is_confirmed_patient_condition
from clingrounder.kg.validator import KGValidator
from clingrounder.schema.annotation import CandidateConcept
from clingrounder.schema.annotation import EntityAnnotation, RelationAnnotation
from clingrounder.schema.types import AssertionStatus, CodeSystem, EntityType, RelationType


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


def test_ontology_code_systems_are_type_constrained() -> None:
    assert code_system_valid_for_entity_type(EntityType.DISEASE, CodeSystem.MONDO)
    assert not code_system_valid_for_entity_type(EntityType.DRUG, CodeSystem.MONDO)
    assert code_system_valid_for_entity_type(EntityType.FINDING, CodeSystem.HPO)
    assert code_system_valid_for_entity_type(EntityType.SYMPTOM, CodeSystem.HPO)
    assert not code_system_valid_for_entity_type(EntityType.DISEASE, CodeSystem.HPO)

    assert CodeSystem.MONDO in (allowed_code_systems(EntityType.DISEASE) or ())
    assert CodeSystem.HPO in (allowed_code_systems(EntityType.FINDING) or ())


def test_kg_validator_reports_without_deleting_invalid_assignment() -> None:
    entity = EntityAnnotation(
        id="E1",
        span=(0, 9),
        text="metformin",
        normalized_text="metformin",
        type=EntityType.DRUG,
        assertion=AssertionStatus.PRESENT,
        code_system=CodeSystem.ICD10,
        code="E11",
        candidates=[
            CandidateConcept(
                code_system=CodeSystem.ICD10,
                code="E11",
                name="Type 2 diabetes mellitus",
                retrieval_score=0.9,
                emit_probability=0.9,
                concept_id="ICD-10:E11",
                source="test",
                evidence_sources=("test",),
                matched_alias="metformin",
                qualified=True,
                qualification_reason="test_candidate",
            )
        ],
    )

    entities, issues = KGValidator().validate_entities([entity])

    assert [issue.kind for issue in issues] == ["invalid_code_system"]
    assert entities[0].code_system == CodeSystem.ICD10
    assert entities[0].code == "E11"
    assert len(entities[0].candidates) == 1
    assert not entity_code_system_valid(entities[0])


def test_treats_requires_drug_head() -> None:
    drug = EntityAnnotation("E1", (0, 9), "metformin", "metformin", EntityType.DRUG)
    disease = EntityAnnotation(
        "E2", (20, 41), "đái tháo đường type 2", "đái tháo đường type 2", EntityType.DISEASE
    )
    relation = RelationAnnotation("R1", "E1", "E2", RelationType.TREATS, 0.9)
    assert relation_type_valid(relation, {"E1": drug, "E2": disease})


def test_has_test_rejects_drug_to_disease_pair() -> None:
    drug = EntityAnnotation("E1", (0, 9), "metformin", "metformin", EntityType.DRUG)
    disease = EntityAnnotation(
        "E2", (20, 41), "đái tháo đường type 2", "đái tháo đường type 2", EntityType.DISEASE
    )
    relation = RelationAnnotation("R1", "E1", "E2", RelationType.HAS_TEST, 0.9)

    assert not relation_type_valid(relation, {"E1": drug, "E2": disease})


def test_has_dose_requires_dedicated_medication_attribute() -> None:
    drug = EntityAnnotation("E1", (0, 9), "metformin", "metformin", EntityType.DRUG)
    strength = EntityAnnotation("E2", (10, 14), "25mg", "25mg", EntityType.STRENGTH)
    lab_result = EntityAnnotation("E3", (20, 23), "120", "120", EntityType.LAB_RESULT)

    valid = RelationAnnotation("R1", "E1", "E2", RelationType.HAS_DOSE, 0.9)
    invalid = RelationAnnotation("R2", "E1", "E3", RelationType.HAS_DOSE, 0.9)

    entities = {entity.id: entity for entity in (drug, strength, lab_result)}
    assert relation_type_valid(valid, entities)
    assert not relation_type_valid(invalid, entities)


def test_unknown_relation_type_is_rejected() -> None:
    disease = EntityAnnotation("E1", (0, 9), "viêm phổi", "viêm phổi", EntityType.DISEASE)
    symptom = EntityAnnotation("E2", (20, 22), "ho", "ho", EntityType.SYMPTOM)
    relation = RelationAnnotation("R1", "E1", "E2", RelationType.UNKNOWN, 0.1)

    assert not relation_type_valid(relation, {"E1": disease, "E2": symptom})


def test_only_present_assertion_is_confirmed_patient_condition() -> None:
    present = EntityAnnotation(
        "E1", (0, 5), "dummy", "dummy", EntityType.DISEASE, assertion=AssertionStatus.PRESENT
    )
    historical = EntityAnnotation(
        "E2",
        (10, 15),
        "dummy",
        "dummy",
        EntityType.DISEASE,
        assertion=AssertionStatus.HISTORICAL,
    )
    possible = EntityAnnotation(
        "E3", (20, 25), "dummy", "dummy", EntityType.DISEASE, assertion=AssertionStatus.POSSIBLE
    )
    family = EntityAnnotation(
        "E4", (30, 35), "dummy", "dummy", EntityType.DISEASE, assertion=AssertionStatus.FAMILY
    )
    negated = EntityAnnotation(
        "E5", (40, 45), "dummy", "dummy", EntityType.DISEASE, assertion=AssertionStatus.NEGATED
    )

    assert is_confirmed_patient_condition(present)
    assert not is_confirmed_patient_condition(historical)
    assert not is_confirmed_patient_condition(possible)
    assert not is_confirmed_patient_condition(family)
    assert not is_confirmed_patient_condition(negated)
