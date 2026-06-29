from __future__ import annotations
from medical_kg_nlp.schema.annotation import EntityAnnotation, RelationAnnotation
from medical_kg_nlp.schema.types import CodeSystem, EntityType, RelationType


def entity_code_system_valid(entity: EntityAnnotation) -> bool:
    if entity.type == EntityType.DRUG:
        return entity.code_system in {CodeSystem.RXNORM, CodeSystem.NONE}
    if entity.type == EntityType.DISEASE:
        return entity.code_system in {CodeSystem.ICD10, CodeSystem.UMLS, CodeSystem.SNOMED, CodeSystem.NONE}
    if entity.type == EntityType.SYMPTOM:
        return entity.code_system in {CodeSystem.UMLS, CodeSystem.SNOMED, CodeSystem.LOCAL, CodeSystem.NONE}
    if entity.type == EntityType.LAB_RESULT:
        return entity.code_system in {CodeSystem.NONE, CodeSystem.LOCAL}
    return True


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
    if relation.type == RelationType.SUGGESTS:
        return head.type in {EntityType.LAB_TEST, EntityType.FINDING} and tail.type in {EntityType.DISEASE, EntityType.FINDING}
    return True

