from __future__ import annotations
from medical_kg_nlp.schema.annotation import EntityAnnotation


def deduplicate_spans(entities: list[EntityAnnotation]) -> list[EntityAnnotation]:
    seen: set[tuple[int, int, str]] = set()
    result: list[EntityAnnotation] = []
    for entity in sorted(entities, key=lambda item: (item.span[0], -(item.span[1] - item.span[0]))):
        key = (entity.span[0], entity.span[1], entity.type.value)
        if key in seen:
            continue
        seen.add(key)
        result.append(entity)
    return result

