from __future__ import annotations
from medical_kg_nlp.schema.annotation import EntityAnnotation, RelationAnnotation
from medical_kg_nlp.schema.types import CodeSystem, EntityType, RelationType


ENTITY_CODE_SYSTEMS: dict[EntityType, frozenset[CodeSystem]] = {
    EntityType.DISEASE: frozenset(
        {CodeSystem.ICD10, CodeSystem.UMLS, CodeSystem.SNOMED, CodeSystem.NONE}
    ),
    EntityType.SYMPTOM: frozenset(
        {CodeSystem.UMLS, CodeSystem.SNOMED, CodeSystem.LOCAL, CodeSystem.NONE}
    ),
    EntityType.DRUG: frozenset({CodeSystem.RXNORM, CodeSystem.NONE}),
    EntityType.LAB_TEST: frozenset({CodeSystem.LOCAL, CodeSystem.NONE}),
    EntityType.LAB_RESULT: frozenset({CodeSystem.LOCAL, CodeSystem.NONE}),
    EntityType.PROCEDURE: frozenset(
        {CodeSystem.ICD10, CodeSystem.UMLS, CodeSystem.SNOMED, CodeSystem.LOCAL, CodeSystem.NONE}
    ),
    EntityType.PATIENT_INFO: frozenset({CodeSystem.LOCAL, CodeSystem.NONE}),
    EntityType.ANATOMY: frozenset({CodeSystem.UMLS, CodeSystem.SNOMED, CodeSystem.LOCAL, CodeSystem.NONE}),
    EntityType.FINDING: frozenset({CodeSystem.UMLS, CodeSystem.SNOMED, CodeSystem.LOCAL, CodeSystem.NONE}),
    EntityType.OTHER: frozenset({CodeSystem.LOCAL, CodeSystem.NONE}),
}


def code_system_valid_for_entity_type(entity_type: EntityType, code_system: CodeSystem) -> bool:
    return code_system in ENTITY_CODE_SYSTEMS[entity_type]


def entity_code_system_valid(entity: EntityAnnotation) -> bool:
    return code_system_valid_for_entity_type(entity.type, entity.code_system)


def relation_type_valid(relation: RelationAnnotation, entities_by_id: dict[str, EntityAnnotation]) -> bool:
    head = entities_by_id.get(relation.head)
    tail = entities_by_id.get(relation.tail)
    if head is None or tail is None:
        return False
    if relation.type == RelationType.TREATS:
        return head.type == EntityType.DRUG and tail.type in {EntityType.DISEASE, EntityType.SYMPTOM}
    if relation.type == RelationType.HAS_SYMPTOM:
        return head.type == EntityType.DISEASE and tail.type == EntityType.SYMPTOM
    if relation.type == RelationType.HAS_VALUE:
        return head.type == EntityType.LAB_TEST and tail.type == EntityType.LAB_RESULT
    if relation.type == RelationType.HAS_DOSE:
        return head.type == EntityType.DRUG and tail.type == EntityType.LAB_RESULT
    if relation.type == RelationType.HAS_ROUTE:
        return head.type == EntityType.DRUG and tail.type == EntityType.LAB_RESULT
    if relation.type == RelationType.HAS_FREQUENCY:
        return head.type == EntityType.DRUG and tail.type == EntityType.LAB_RESULT
    if relation.type == RelationType.SUGGESTS:
        return head.type in {EntityType.LAB_TEST, EntityType.FINDING} and tail.type in {EntityType.DISEASE, EntityType.FINDING}
    if relation.type == RelationType.HAS_TEST:
        return head.type in {EntityType.DISEASE, EntityType.FINDING} and tail.type == EntityType.LAB_TEST
    if relation.type == RelationType.CAUSED_BY:
        return head.type in {EntityType.DISEASE, EntityType.SYMPTOM, EntityType.FINDING} and tail.type in {
            EntityType.DISEASE,
            EntityType.DRUG,
            EntityType.FINDING,
            EntityType.PROCEDURE,
        }
    if relation.type == RelationType.ASSOCIATED_WITH:
        clinical_types = {
            EntityType.DISEASE,
            EntityType.SYMPTOM,
            EntityType.DRUG,
            EntityType.LAB_TEST,
            EntityType.LAB_RESULT,
            EntityType.PROCEDURE,
            EntityType.ANATOMY,
            EntityType.FINDING,
        }
        return head.type in clinical_types and tail.type in clinical_types
    if relation.type == RelationType.IS_A:
        return head.type == tail.type and head.type not in {EntityType.PATIENT_INFO, EntityType.OTHER}
    if relation.type == RelationType.PART_OF:
        return head.type == EntityType.ANATOMY and tail.type == EntityType.ANATOMY
    if relation.type == RelationType.NEGATES:
        return head.type == EntityType.OTHER and tail.type in {
            EntityType.DISEASE,
            EntityType.SYMPTOM,
            EntityType.FINDING,
        }
    if relation.type == RelationType.UNKNOWN:
        return False
    return False
