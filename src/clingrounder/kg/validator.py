from __future__ import annotations
from dataclasses import dataclass

from clingrounder.kg.constraints import entity_code_system_valid, relation_type_valid
from clingrounder.kg.ontology_reasoner import OntologyReasoner
from clingrounder.schema.annotation import EntityAnnotation, RelationAnnotation
from clingrounder.schema.types import RelationType


@dataclass(frozen=True)
class ValidationIssue:
    kind: str
    target_id: str
    message: str


class KGValidator:
    """Check KG constraints without owning terminology storage or mutating entities."""

    def __init__(self, reasoner: OntologyReasoner | None = None) -> None:
        self.reasoner = reasoner

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
                # INVARIANT: final validation must observe the original invalid assignment.
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
            if not relation_type_valid(relation, by_id):
                issues.append(
                    ValidationIssue(
                        kind="invalid_relation",
                        target_id=relation.id,
                        message=f"Invalid {relation.type.value} relation between {relation.head} and {relation.tail}",
                    )
                )
                continue
            ontology_status = self._ontology_relation_status(relation, by_id)
            if ontology_status == "valid":
                valid.append(relation)
            else:
                issues.append(
                    ValidationIssue(
                        kind=ontology_status,
                        target_id=relation.id,
                        message=(
                            f"Cannot validate {relation.type.value} between "
                            f"{relation.head} and {relation.tail}: {ontology_status}"
                        ),
                    )
                )
        return valid, issues

    def _ontology_relation_status(
        self,
        relation: RelationAnnotation,
        entities_by_id: dict[str, EntityAnnotation],
    ) -> str:
        child = entities_by_id[relation.head]
        parent = entities_by_id[relation.tail]
        if self.reasoner is not None:
            for entity in (child, parent):
                if entity.code is not None and not self.reasoner.contains(
                    entity.code_system, entity.code
                ):
                    return "unknown_ontology_membership"
        if relation.type != RelationType.IS_A:
            return "valid"
        if self.reasoner is None:
            return "unknown_ontology_membership"
        if (
            child.code is None
            or parent.code is None
            or child.code_system != parent.code_system
            or not self.reasoner.contains(child.code_system, child.code)
            or not self.reasoner.contains(parent.code_system, parent.code)
        ):
            return "unknown_ontology_membership"
        if not self.reasoner.is_a(child.code_system, child.code, parent.code):
            return "invalid_relation"
        return "valid"
