from __future__ import annotations
from clingrounder.schema.annotation import EntityAnnotation


def nearby_pairs(entities: list[EntityAnnotation], max_distance: int = 160) -> list[tuple[EntityAnnotation, EntityAnnotation]]:
    pairs: list[tuple[EntityAnnotation, EntityAnnotation]] = []
    for left_index, left in enumerate(entities):
        for right in entities[left_index + 1 :]:
            if right.span[0] - left.span[1] <= max_distance:
                pairs.append((left, right))
    return pairs

