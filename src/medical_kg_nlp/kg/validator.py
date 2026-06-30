from __future__ import annotations
from dataclasses import dataclass

from medical_kg_nlp.kg.constraints import entity_code_system_valid, relation_type_valid
from medical_kg_nlp.schema.annotation import EntityAnnotation, RelationAnnotation
from medical_kg_nlp.schema.types import CodeSystem


@dataclass(frozen=True)
class ValidationIssue:
    kind: str
    target_id: str
    message: str


class KGValidator:
    def validate_entities(self, entities: list[EntityAnnotation]) -> tuple[list[EntityAnnotation], list[ValidationIssue]]:
        valid: list[EntityAnnotation] = []
        issues: list[ValidationIssue] = []
        for entity in entities:
            if entity_code_system_valid(entity):
                valid.append(entity)
            else:
                issues.append(
                    ValidationIssue(
                        kind="invalid_code_system",
                        target_id=entity.id,
                        message=f"{entity.type.value} cannot map to {entity.code_system.value}",
                    )
                )
                entity.code = None
                entity.code_system = CodeSystem.NONE
                entity.candidates = []
                valid.append(entity)
        return valid, issues

    def validate_relations(
        self,
        entities: list[EntityAnnotation],
        relations: list[RelationAnnotation],
    ) -> tuple[list[RelationAnnotation], list[ValidationIssue]]:
        by_id = {entity.id: entity for entity in entities}
        valid: list[RelationAnnotation] = []
        issues: list[ValidationIssue] = []
        for relation in relations:
            if relation_type_valid(relation, by_id):
                valid.append(relation)
            else:
                issues.append(
                    ValidationIssue(
                        kind="invalid_relation",
                        target_id=relation.id,
                        message=f"Invalid {relation.type.value} relation between {relation.head} and {relation.tail}",
                    )
                )
        return valid, issues
