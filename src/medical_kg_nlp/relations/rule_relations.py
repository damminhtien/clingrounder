from __future__ import annotations
from medical_kg_nlp.schema.annotation import EntityAnnotation, RelationAnnotation
from medical_kg_nlp.schema.document import Sentence
from medical_kg_nlp.schema.types import AssertionStatus, EntityType, RelationType


KNOWN_TREATS = {
    ("amoxicillin", "J18.9"),
    ("aspirin", "I21.9"),
    ("atorvastatin", "I25.10"),
    ("lisinopril", "I10"),
    ("losartan", "I10"),
    ("metformin", "E11"),
    ("omeprazole", "K21.9"),
    ("salbutamol", "J45"),
}


class RuleRelationExtractor:
    def extract(
        self,
        entities: list[EntityAnnotation],
        sentences: list[Sentence],
    ) -> list[RelationAnnotation]:
        relations: list[RelationAnnotation] = []
        relations.extend(self._medication_attribute_relations(entities, sentences))
        relations.extend(self._lab_value_relations(entities, sentences))
        relations.extend(self._sentence_relations(entities, sentences))
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

    def _medication_attribute_relations(
        self,
        entities: list[EntityAnnotation],
        sentences: list[Sentence],
    ) -> list[RelationAnnotation]:
        relations: list[RelationAnnotation] = []
        relation_by_type = {
            EntityType.DOSAGE: RelationType.HAS_DOSE,
            EntityType.STRENGTH: RelationType.HAS_DOSE,
            EntityType.ROUTE: RelationType.HAS_ROUTE,
            EntityType.FREQUENCY: RelationType.HAS_FREQUENCY,
            EntityType.DURATION: RelationType.HAS_DURATION,
            EntityType.DOSAGE_FORM: RelationType.HAS_DOSAGE_FORM,
        }
        for sentence in sentences:
            inside = self._inside_sentence(entities, sentence)
            drugs = [
                entity for entity in inside if entity.type == EntityType.DRUG and _active(entity)
            ]
            attributes = [
                entity for entity in inside if entity.type in relation_by_type and _active(entity)
            ]
            for attribute in attributes:
                drug = _nearest(attribute, drugs, max_distance=80)
                if drug is None:
                    continue
                relations.append(
                    RelationAnnotation(
                        id="",
                        head=drug.id,
                        tail=attribute.id,
                        type=relation_by_type[attribute.type],
                        confidence=0.9,
                        evidence_span=(
                            min(drug.span[0], attribute.span[0]),
                            max(drug.span[1], attribute.span[1]),
                        ),
                    )
                )
        return relations

    def _lab_value_relations(
        self,
        entities: list[EntityAnnotation],
        sentences: list[Sentence],
    ) -> list[RelationAnnotation]:
        relations: list[RelationAnnotation] = []
        for sentence in sentences:
            inside = self._inside_sentence(entities, sentence)
            tests = [
                entity
                for entity in inside
                if entity.type == EntityType.LAB_TEST and _active(entity)
            ]
            results = [
                entity
                for entity in inside
                if entity.type == EntityType.LAB_RESULT and _active(entity)
            ]
            for result in results:
                test = _nearest(result, tests, max_distance=80)
                if test is None:
                    continue
                relations.append(
                    RelationAnnotation(
                        id="",
                        head=test.id,
                        tail=result.id,
                        type=RelationType.HAS_VALUE,
                        confidence=0.85,
                        evidence_span=(
                            min(test.span[0], result.span[0]),
                            max(test.span[1], result.span[1]),
                        ),
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
            inside = self._inside_sentence(entities, sentence)
            diseases = [
                entity for entity in inside if entity.type == EntityType.DISEASE and _active(entity)
            ]
            symptoms = [
                entity for entity in inside if entity.type == EntityType.SYMPTOM and _active(entity)
            ]
            tests = [
                entity
                for entity in inside
                if entity.type == EntityType.LAB_TEST and _active(entity)
            ]
            for symptom in symptoms:
                disease = _nearest(symptom, diseases, max_distance=160)
                if disease is None:
                    continue
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
                disease = _nearest(test, diseases, max_distance=160)
                if disease is None:
                    continue
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

    def _known_treatment_relations(
        self, entities: list[EntityAnnotation]
    ) -> list[RelationAnnotation]:
        relations: list[RelationAnnotation] = []
        drugs = [entity for entity in entities if entity.type == EntityType.DRUG]
        diseases = [entity for entity in entities if entity.type == EntityType.DISEASE]
        for drug in drugs:
            for disease in diseases:
                if not _active(drug) or not _active(disease):
                    continue
                if (drug.normalized_text, disease.code or "") in KNOWN_TREATS:
                    relations.append(
                        RelationAnnotation(
                            id="",
                            head=drug.id,
                            tail=disease.id,
                            type=RelationType.TREATS,
                            confidence=0.75,
                            evidence_span=(
                                min(drug.span[0], disease.span[0]),
                                max(drug.span[1], disease.span[1]),
                            ),
                        )
                    )
        return relations

    @staticmethod
    def _inside_sentence(
        entities: list[EntityAnnotation],
        sentence: Sentence,
    ) -> list[EntityAnnotation]:
        return [
            entity
            for entity in entities
            if sentence.span[0] <= entity.span[0] and entity.span[1] <= sentence.span[1]
        ]


def _active(entity: EntityAnnotation) -> bool:
    return entity.assertion not in {AssertionStatus.NEGATED, AssertionStatus.FAMILY}


def _nearest(
    target: EntityAnnotation,
    candidates: list[EntityAnnotation],
    *,
    max_distance: int,
) -> EntityAnnotation | None:
    ranked = sorted(
        (
            (_span_distance(target.span, candidate.span), candidate.span[0], candidate)
            for candidate in candidates
        ),
        key=lambda item: (item[0], item[1]),
    )
    if not ranked or ranked[0][0] > max_distance:
        return None
    return ranked[0][2]


def _span_distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    if left[1] <= right[0]:
        return right[0] - left[1]
    if right[1] <= left[0]:
        return left[0] - right[1]
    return 0
