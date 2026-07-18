"""Deterministic composition of model recall and reviewed dictionary NER."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from medical_kg_nlp.schema.annotation import EntityAnnotation

if TYPE_CHECKING:
    from medical_kg_nlp.pipeline.ports import EntityExtractorPort

__all__ = ["HybridEntityExtractorAdapter"]


@dataclass(frozen=True)
class HybridEntityExtractorAdapter:
    """Keep reviewed dictionary spans and add non-overlapping model proposals.

    The dictionary side is authoritative for overlap conflicts; the model supplies recall
    for concepts not yet present in the controlled recognition dictionary.
    """

    model: EntityExtractorPort
    dictionary: EntityExtractorPort

    def extract(self, source_text: str) -> list[EntityAnnotation]:
        dictionary_entities = self._validated(self.dictionary.extract(source_text), source_text)
        model_entities = self._validated(self.model.extract(source_text), source_text)
        selected: list[EntityAnnotation] = list(
            sorted(dictionary_entities, key=_entity_order)
        )

        # MODEL: model spans are recall proposals only; a reviewed dictionary overlap wins,
        # including a same-span type conflict. This prevents model noise replacing a known term.
        for entity in sorted(model_entities, key=_model_priority):
            if any(_overlaps(entity, existing) for existing in selected):
                continue
            selected.append(entity)

        selected.sort(key=_entity_order)
        # INVARIANT: IDs are regenerated after fusion so downstream relations see unique IDs.
        return [replace(entity, id=f"H{index:04d}") for index, entity in enumerate(selected, 1)]

    @staticmethod
    def _validated(
        entities: list[EntityAnnotation],
        source_text: str,
    ) -> list[EntityAnnotation]:
        for entity in entities:
            entity.validate_offsets(source_text)
        return entities


def _overlaps(left: EntityAnnotation, right: EntityAnnotation) -> bool:
    return left.span[0] < right.span[1] and right.span[0] < left.span[1]


def _entity_order(entity: EntityAnnotation) -> tuple[int, int, str, str]:
    return (entity.span[0], entity.span[1], entity.type.value, entity.id)


def _model_priority(entity: EntityAnnotation) -> tuple[float, int, int, str, str]:
    start, end = entity.span
    return (-entity.confidence, -(end - start), start, entity.type.value, entity.id)
