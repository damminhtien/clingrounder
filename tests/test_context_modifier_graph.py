"""Context graph tests protect evidence provenance and raw offsets."""

import pytest

from clingrounder.context import AssertionClassifier
from clingrounder.schema.annotation import EntityAnnotation
from clingrounder.schema.document import Sentence
from clingrounder.schema.types import CodeSystem, EntityType


def test_context_graph_reuses_one_modifier_for_coordinated_targets() -> None:
    text = "Bệnh nhân không chóng mặt, buồn nôn."
    sentence = Sentence(span=(0, len(text)), text=text)
    entities = [
        _entity(text, "chóng mặt", EntityType.SYMPTOM),
        _entity(text, "buồn nôn", EntityType.SYMPTOM),
    ]

    decisions, graph = AssertionClassifier().classify_batch_with_graph(
        entities,
        sentence,
    )

    assert all(decisions[entity.id][0].negated for entity in entities)
    assert len(graph.targets) == 2
    assert len(graph.modifiers) == 1
    assert len(graph.edges) == 2
    modifier = graph.modifiers[0]
    assert modifier.span is not None
    assert text[slice(*modifier.span)] == "không"
    assert {edge.target_id for edge in graph.edges} == {entity.id for entity in entities}


def test_context_graph_keeps_section_prior_as_spanless_evidence() -> None:
    text = "Tăng huyết áp."
    sentence = Sentence(
        span=(0, len(text)),
        text=text,
        section_title="Tiền sử bệnh",
    )
    entity = _entity(text, "Tăng huyết áp", EntityType.DISEASE)

    decisions, graph = AssertionClassifier().classify_batch_with_graph(
        [entity],
        sentence,
    )

    assert decisions[entity.id][0].historical is True
    assert len(graph.modifiers) == 1
    assert graph.modifiers[0].span is None
    assert graph.modifiers[0].scope == "section_prior"
    assert graph.edges[0].distance == 0


def test_context_graph_rejects_target_outside_sentence() -> None:
    sentence = Sentence(span=(10, 20), text="0123456789")
    entity = EntityAnnotation(
        id="outside",
        span=(0, 4),
        text="test",
        normalized_text="test",
        type=EntityType.DISEASE,
        code_system=CodeSystem.NONE,
    )

    with pytest.raises(ValueError, match="outside sentence"):
        AssertionClassifier().classify_batch_with_graph([entity], sentence)


def _entity(text: str, mention: str, entity_type: EntityType) -> EntityAnnotation:
    start = text.index(mention)
    return EntityAnnotation(
        id=f"{entity_type.value}:{start}",
        span=(start, start + len(mention)),
        text=mention,
        normalized_text=mention.casefold(),
        type=entity_type,
        code_system=CodeSystem.NONE,
    )
