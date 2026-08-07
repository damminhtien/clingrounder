"""Conservative, data-driven relation extraction baseline.

This module is intentionally a baseline adapter, not a general ontology reasoner. Structural
relations are deterministic; semantic relations require an explicit local cue or a reviewed
code-to-code record. Every emitted relation carries evidence and provenance.
"""

from __future__ import annotations

import re
from pathlib import Path

from clingrounder.relations.knowledge import KnownRelationRepository
from clingrounder.schema.annotation import EntityAnnotation, RelationAnnotation, RelationEvidence
from clingrounder.schema.document import Sentence
from clingrounder.schema.types import AssertionStatus, CodeSystem, EntityType, RelationType

__all__ = ["RuleRelationExtractor"]

_DEFAULT_KNOWN_RELATIONS = Path("data/relations/known_treats.jsonl")
_BLOCKED_ASSERTIONS = frozenset(
    {
        AssertionStatus.NEGATED,
        AssertionStatus.FAMILY,
        AssertionStatus.POSSIBLE,
        AssertionStatus.CONDITIONAL,
        AssertionStatus.PLANNED,
        AssertionStatus.RESOLVED,
    }
)
_RELATION_CUES = re.compile(
    r"\b(?:gây|kèm|vì|do|biểu hiện|triệu chứng|liên quan|associated with|due to|with|causes)\b",
    re.IGNORECASE,
)


class RuleRelationExtractor:
    """Emit only relations supported by local structure, cues, or reviewed codes."""

    def __init__(self, known_relations_path: str | Path = _DEFAULT_KNOWN_RELATIONS) -> None:
        self.known_relations = KnownRelationRepository(known_relations_path)

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
        relation_by_type = {
            EntityType.DOSAGE: RelationType.HAS_DOSE,
            EntityType.STRENGTH: RelationType.HAS_DOSE,
            EntityType.ROUTE: RelationType.HAS_ROUTE,
            EntityType.FREQUENCY: RelationType.HAS_FREQUENCY,
            EntityType.DURATION: RelationType.HAS_DURATION,
            EntityType.DOSAGE_FORM: RelationType.HAS_DOSAGE_FORM,
        }
        relations: list[RelationAnnotation] = []
        for sentence in sentences:
            inside = self._inside_sentence(entities, sentence)
            drugs = [
                entity for entity in inside if entity.type == EntityType.DRUG and _active(entity)
            ]
            attributes = [
                entity
                for entity in inside
                if entity.type in relation_by_type and _active(entity)
            ]
            for attribute in attributes:
                same_clause_drugs = [
                    drug for drug in drugs if _same_clause(drug, attribute, sentence)
                ]
                drug = _nearest(attribute, same_clause_drugs, max_distance=80)
                if drug is None:
                    continue
                evidence_span = _covering_span(drug, attribute)
                relations.append(
                    _relation(
                        head=drug.id,
                        tail=attribute.id,
                        relation_type=relation_by_type[attribute.type],
                        evidence_span=evidence_span,
                        source="structural_medication_attribute",
                        rule_id="structural.same_clause.nearest_attribute",
                        support_score=1.0,
                        provenance="deterministic_same_clause_structure",
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
                entity for entity in inside if entity.type == EntityType.LAB_TEST and _active(entity)
            ]
            results = [
                entity
                for entity in inside
                if entity.type == EntityType.LAB_RESULT and _active(entity)
            ]
            for result in results:
                same_clause_tests = [
                    test for test in tests if _same_clause(test, result, sentence)
                ]
                test = _nearest(result, same_clause_tests, max_distance=80)
                if test is None:
                    continue
                evidence_span = _covering_span(test, result)
                relations.append(
                    _relation(
                        head=test.id,
                        tail=result.id,
                        relation_type=RelationType.HAS_VALUE,
                        evidence_span=evidence_span,
                        source="structural_lab_result",
                        rule_id="structural.same_clause.lab_test_result",
                        support_score=1.0,
                        provenance="deterministic_lab_anchor",
                    )
                )
        return relations

    def _sentence_relations(
        self,
        entities: list[EntityAnnotation],
        sentences: list[Sentence],
    ) -> list[RelationAnnotation]:
        """Emit cue-backed semantic proposals; proximity alone is deliberately insufficient."""

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
                entity for entity in inside if entity.type == EntityType.LAB_TEST and _active(entity)
            ]
            for symptom in symptoms:
                candidates = [
                    disease
                    for disease in diseases
                    if _same_clause(disease, symptom, sentence)
                    and _has_relation_cue(disease, symptom, sentence)
                ]
                disease = _nearest(symptom, candidates, max_distance=160)
                if disease is not None:
                    evidence_span = _covering_span(disease, symptom)
                    relations.append(
                        _relation(
                            head=disease.id,
                            tail=symptom.id,
                            relation_type=RelationType.HAS_SYMPTOM,
                            evidence_span=evidence_span,
                            source="sentence_cooccurrence_proposal",
                            rule_id="semantic.same_clause.explicit_cue",
                            support_score=0.75,
                            provenance="heuristic_score_not_calibrated_probability",
                        )
                    )
            for test in tests:
                candidates = [
                    disease
                    for disease in diseases
                    if _same_clause(disease, test, sentence)
                    and _has_relation_cue(test, disease, sentence)
                ]
                disease = _nearest(test, candidates, max_distance=160)
                if disease is None:
                    continue
                evidence_span = _covering_span(test, disease)
                if _suggestive_cue(sentence.text):
                    relations.append(
                        _relation(
                            head=test.id,
                            tail=disease.id,
                            relation_type=RelationType.SUGGESTS,
                            evidence_span=evidence_span,
                            source="sentence_cooccurrence_proposal",
                            rule_id="semantic.same_clause.suggestive_cue",
                            support_score=0.8,
                            provenance="heuristic_score_not_calibrated_probability",
                        )
                    )
                else:
                    relations.append(
                        _relation(
                            head=disease.id,
                            tail=test.id,
                            relation_type=RelationType.HAS_TEST,
                            evidence_span=evidence_span,
                            source="sentence_cooccurrence_proposal",
                            rule_id="semantic.same_clause.test_cue",
                            support_score=0.7,
                            provenance="heuristic_score_not_calibrated_probability",
                        )
                    )
        return relations

    def _known_treatment_relations(
        self, entities: list[EntityAnnotation]
    ) -> list[RelationAnnotation]:
        relations: list[RelationAnnotation] = []
        drugs = [entity for entity in entities if entity.type == EntityType.DRUG and _active(entity)]
        diseases = [
            entity for entity in entities if entity.type == EntityType.DISEASE and _active(entity)
        ]
        for drug in drugs:
            if drug.code is None or drug.code_system != CodeSystem.RXNORM:
                continue
            for disease in diseases:
                if disease.code is None or disease.code_system != CodeSystem.ICD10:
                    continue
                known = self.known_relations.find(
                    CodeSystem.RXNORM,
                    drug.code,
                    RelationType.TREATS,
                    CodeSystem.ICD10,
                    disease.code,
                )
                if known is None:
                    continue
                evidence_span = _covering_span(drug, disease)
                relations.append(
                    _relation(
                        head=drug.id,
                        tail=disease.id,
                        relation_type=RelationType.TREATS,
                        evidence_span=evidence_span,
                        source="terminology_ontology_backed",
                        rule_id="known_treats.reviewed_code_pair",
                        support_score=1.0,
                        provenance=(
                            f"{known.source}@{known.source_version};"
                            f"review_status={known.review_status}"
                        ),
                    )
                )
        return relations

    @staticmethod
    def _inside_sentence(
        entities: list[EntityAnnotation], sentence: Sentence
    ) -> list[EntityAnnotation]:
        return [
            entity
            for entity in entities
            if sentence.span[0] <= entity.span[0] and entity.span[1] <= sentence.span[1]
        ]


def _relation(
    *,
    head: str,
    tail: str,
    relation_type: RelationType,
    evidence_span: tuple[int, int],
    source: str,
    rule_id: str,
    support_score: float,
    provenance: str,
) -> RelationAnnotation:
    return RelationAnnotation(
        id=f"relation:{head}:{tail}:{relation_type.value}:{evidence_span[0]}:{evidence_span[1]}",
        head=head,
        tail=tail,
        type=relation_type,
        confidence=support_score,
        evidence_span=evidence_span,
        evidence=RelationEvidence(
            source=source,
            rule_id=rule_id,
            evidence_span=evidence_span,
            support_score=support_score,
            provenance=provenance,
        ),
    )


def _active(entity: EntityAnnotation) -> bool:
    return entity.assertion not in _BLOCKED_ASSERTIONS


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


def _same_clause(
    left: EntityAnnotation, right: EntityAnnotation, sentence: Sentence
) -> bool:
    start = min(left.span[1], right.span[1])
    end = max(left.span[0], right.span[0])
    between = sentence.text[start - sentence.span[0] : end - sentence.span[0]]
    return not bool(re.search(r"[\n;.!?]|(?:^|\n)\s*(?:[-*]|\d+[.)])\s*", between))


def _has_relation_cue(
    left: EntityAnnotation, right: EntityAnnotation, sentence: Sentence
) -> bool:
    start = min(left.span[1], right.span[1])
    end = max(left.span[0], right.span[0])
    between = sentence.text[start - sentence.span[0] : end - sentence.span[0]]
    return bool(_RELATION_CUES.search(between))


def _suggestive_cue(text: str) -> bool:
    return bool(re.search(r"\b(?:gợi ý|suggest(?:s|ive)?|phù hợp với)\b", text, re.IGNORECASE))


def _covering_span(left: EntityAnnotation, right: EntityAnnotation) -> tuple[int, int]:
    return min(left.span[0], right.span[0]), max(left.span[1], right.span[1])


def _span_distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    if left[1] <= right[0]:
        return right[0] - left[1]
    if right[1] <= left[0]:
        return left[0] - right[1]
    return 0
