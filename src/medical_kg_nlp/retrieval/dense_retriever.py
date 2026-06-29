from __future__ import annotations
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.schema.types import EntityType


class DenseRetriever:
    """Optional dense retriever placeholder. Keep final codes dictionary constrained."""

    def retrieve(self, mention: str, entity_type: EntityType | None = None, limit: int = 20) -> list[Candidate]:
        return []

