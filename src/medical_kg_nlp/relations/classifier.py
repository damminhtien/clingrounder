from __future__ import annotations
from medical_kg_nlp.schema.types import RelationType


class RelationClassifier:
    """Future entity-marker transformer relation classifier."""

    def predict(self, marked_sentence: str) -> RelationType:
        return RelationType.UNKNOWN

