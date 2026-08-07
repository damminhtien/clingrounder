"""Independent relation evaluation slices for baseline diagnosis."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from clingrounder.schema.annotation import EntityAnnotation, RelationAnnotation
from clingrounder.schema.document import Sentence
from clingrounder.schema.types import EntityType

__all__ = ["relation_slice_counts"]


def relation_slice_counts(
    relations: Iterable[RelationAnnotation],
    entities: Iterable[EntityAnnotation],
    sentences: Iterable[Sentence] = (),
) -> dict[str, dict[str, int]]:
    """Count predictions by relation type, evidence, scope, assertion, and domain.

    The report is diagnostic only. It does not infer correctness from ontology absence and is
    deliberately independent of any competition-specific scorer.
    """

    by_id = {entity.id: entity for entity in entities}
    sentence_list = tuple(sentences)
    report: dict[str, Counter[str]] = {
        "relation_type": Counter(),
        "distance": Counter(),
        "scope": Counter(),
        "assertion_status": Counter(),
        "source": Counter(),
        "evidence_kind": Counter(),
        "domain": Counter(),
    }
    for relation in relations:
        head = by_id.get(relation.head)
        tail = by_id.get(relation.tail)
        report["relation_type"][relation.type.value] += 1
        if head is None or tail is None:
            report["scope"]["unknown_endpoint"] += 1
            continue
        distance = abs(head.span[0] - tail.span[0])
        report["distance"][_distance_bucket(distance)] += 1
        report["scope"][_scope_bucket(head, tail, sentence_list)] += 1
        report["assertion_status"][f"{head.assertion.value}|{tail.assertion.value}"] += 1
        evidence = relation.evidence
        source = evidence.source if evidence is not None else "unknown"
        report["source"][source] += 1
        report["evidence_kind"][
            "terminology_backed"
            if source == "terminology_ontology_backed"
            else "heuristic_or_structural"
        ] += 1
        if _is_medication_relation(head.type, tail.type):
            report["domain"]["medication_list"] += 1
        if _is_lab_relation(head.type, tail.type):
            report["domain"]["lab_list"] += 1
    return {name: dict(sorted(counter.items())) for name, counter in report.items()}


def _distance_bucket(distance: int) -> str:
    if distance <= 40:
        return "0-40"
    if distance <= 160:
        return "41-160"
    return "161+"


def _scope_bucket(
    head: EntityAnnotation,
    tail: EntityAnnotation,
    sentences: tuple[Sentence, ...],
) -> str:
    head_sentence = next((item for item in sentences if _contains(item, head)), None)
    tail_sentence = next((item for item in sentences if _contains(item, tail)), None)
    if head_sentence is None or tail_sentence is None:
        return "unknown"
    if head_sentence.span != tail_sentence.span:
        return "cross_sentence"
    start = min(head.span[1], tail.span[1]) - head_sentence.span[0]
    end = max(head.span[0], tail.span[0]) - head_sentence.span[0]
    between = head_sentence.text[start:end]
    return "same_clause" if not any(char in between for char in "\n;.!?") else "same_sentence"


def _contains(sentence: Sentence, entity: EntityAnnotation) -> bool:
    return sentence.span[0] <= entity.span[0] and entity.span[1] <= sentence.span[1]


def _is_medication_relation(head_type: EntityType, tail_type: EntityType) -> bool:
    return head_type == EntityType.DRUG and tail_type in {
        EntityType.DOSAGE,
        EntityType.STRENGTH,
        EntityType.ROUTE,
        EntityType.FREQUENCY,
        EntityType.DURATION,
        EntityType.DOSAGE_FORM,
    }


def _is_lab_relation(head_type: EntityType, tail_type: EntityType) -> bool:
    return head_type == EntityType.LAB_TEST and tail_type == EntityType.LAB_RESULT
