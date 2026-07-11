from __future__ import annotations
from dataclasses import dataclass

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.kg.constraints import entity_code_system_valid, relation_type_valid
from medical_kg_nlp.kg.ontology_reasoner import OntologyReasoner
from medical_kg_nlp.schema.annotation import EntityAnnotation, RelationAnnotation
from medical_kg_nlp.schema.types import CodeSystem, RelationType


@dataclass(frozen=True)
class ValidationIssue:
    kind: str
    target_id: str
    message: str


class KGValidator:
    def __init__(self, dictionary: DictionaryStore | None = None) -> None:
        self.reasoner = OntologyReasoner(dictionary) if dictionary is not None else None

    def validate_entities(
        self, entities: list[EntityAnnotation]
    ) -> tuple[list[EntityAnnotation], list[ValidationIssue]]:
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
            if relation_type_valid(relation, by_id) and self._ontology_relation_valid(
                relation, by_id
            ):
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

    def _ontology_relation_valid(
        self,
        relation: RelationAnnotation,
        entities_by_id: dict[str, EntityAnnotation],
    ) -> bool:
        if relation.type != RelationType.IS_A or self.reasoner is None:
            return True
        child = entities_by_id[relation.head]
        parent = entities_by_id[relation.tail]
        if (
            child.code is None
            or parent.code is None
            or child.code_system != parent.code_system
            or not self.reasoner.contains(child.code_system, child.code)
            or not self.reasoner.contains(parent.code_system, parent.code)
        ):
            return True
        return self.reasoner.is_a(child.code_system, child.code, parent.code)
