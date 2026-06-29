from __future__ import annotations
from medical_kg_nlp.schema.annotation import EntityAnnotation, RelationAnnotation
from medical_kg_nlp.schema.document import Sentence
from medical_kg_nlp.schema.types import AssertionStatus, EntityType, RelationType


KNOWN_TREATS = {
    ("metformin", "E11"),
    ("salbutamol", "J45"),
}


class RuleRelationExtractor:
    def extract(
        self,
        entities: list[EntityAnnotation],
        sentences: list[Sentence],
    ) -> list[RelationAnnotation]:
        relations: list[RelationAnnotation] = []
        relations.extend(self._dose_relations(entities))
        relations.extend(self._sentence_relations(entities, sentences))
        relations.extend(self._nearby_symptom_relations(entities))
        relations.extend(self._known_treatment_relations(entities))
        deduped: list[RelationAnnotation] = []
        seen: set[tuple[str, str, RelationType]] = set()
        for relation in relations:
            key = (relation.head, relation.tail, relation.type)
            if key in seen:
                continue
            seen.add(key)
            relation.id = f"R{len(deduped) + 1}"
            deduped.append(relation)
        return deduped

    def _dose_relations(self, entities: list[EntityAnnotation]) -> list[RelationAnnotation]:
        relations: list[RelationAnnotation] = []
        drugs = [entity for entity in entities if entity.type == EntityType.DRUG]
        values = [entity for entity in entities if entity.type == EntityType.LAB_RESULT]
        for drug in drugs:
            for value in values:
                if 0 <= value.span[0] - drug.span[1] <= 30:
                    relations.append(
                        RelationAnnotation(
                            id="",
                            head=drug.id,
                            tail=value.id,
                            type=RelationType.HAS_DOSE,
                            confidence=0.85,
                            evidence_span=(drug.span[0], value.span[1]),
                        )
                    )
        return relations

    def _sentence_relations(
        self,
        entities: list[EntityAnnotation],
        sentences: list[Sentence],
    ) -> list[RelationAnnotation]:
        relations: list[RelationAnnotation] = []
        for sentence in sentences:
            inside = [entity for entity in entities if sentence.span[0] <= entity.span[0] and entity.span[1] <= sentence.span[1]]
            diseases = [entity for entity in inside if entity.type == EntityType.DISEASE]
            symptoms = [entity for entity in inside if entity.type == EntityType.SYMPTOM]
            tests = [entity for entity in inside if entity.type == EntityType.LAB_TEST]
            for disease in diseases:
                if disease.assertion in {AssertionStatus.NEGATED, AssertionStatus.FAMILY}:
                    continue
                for symptom in symptoms:
                    relations.append(
                        RelationAnnotation(
                            id="",
                            head=disease.id,
                            tail=symptom.id,
                            type=RelationType.HAS_SYMPTOM,
                            confidence=0.6,
                            evidence_span=sentence.span,
                        )
                    )
                for test in tests:
                    if any(cue in sentence.text.lower() for cue in ("gợi ý", "suggest", "suggestive")):
                        relations.append(
                            RelationAnnotation(
                                id="",
                                head=test.id,
                                tail=disease.id,
                                type=RelationType.SUGGESTS,
                                confidence=0.7,
                                evidence_span=sentence.span,
                            )
                        )
                    else:
                        relations.append(
                            RelationAnnotation(
                                id="",
                                head=disease.id,
                                tail=test.id,
                                type=RelationType.HAS_TEST,
                                confidence=0.55,
                                evidence_span=sentence.span,
                            )
                        )
        return relations

    def _nearby_symptom_relations(self, entities: list[EntityAnnotation]) -> list[RelationAnnotation]:
        relations: list[RelationAnnotation] = []
        diseases = [entity for entity in entities if entity.type == EntityType.DISEASE]
        symptoms = [entity for entity in entities if entity.type == EntityType.SYMPTOM]
        for disease in diseases:
            if disease.assertion in {AssertionStatus.NEGATED, AssertionStatus.FAMILY}:
                continue
            for symptom in symptoms:
                distance = disease.span[0] - symptom.span[1]
                if 0 <= distance <= 80:
                    relations.append(
                        RelationAnnotation(
                            id="",
                            head=disease.id,
                            tail=symptom.id,
                            type=RelationType.HAS_SYMPTOM,
                            confidence=0.55,
                            evidence_span=(symptom.span[0], disease.span[1]),
                        )
                    )
        return relations

    def _known_treatment_relations(self, entities: list[EntityAnnotation]) -> list[RelationAnnotation]:
        relations: list[RelationAnnotation] = []
        drugs = [entity for entity in entities if entity.type == EntityType.DRUG]
        diseases = [entity for entity in entities if entity.type == EntityType.DISEASE]
        for drug in drugs:
            for disease in diseases:
                if disease.assertion in {AssertionStatus.NEGATED, AssertionStatus.FAMILY}:
                    continue
                if (drug.normalized_text, disease.code or "") in KNOWN_TREATS:
                    relations.append(
                        RelationAnnotation(
                            id="",
                            head=drug.id,
                            tail=disease.id,
                            type=RelationType.TREATS,
                            confidence=0.75,
                            evidence_span=(min(drug.span[0], disease.span[0]), max(drug.span[1], disease.span[1])),
                        )
                    )
        return relations
