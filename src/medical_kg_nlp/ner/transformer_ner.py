from __future__ import annotations
from medical_kg_nlp.schema.annotation import EntityAnnotation


class TransformerNER:
    """Future BIO/BIOES token-classification baseline with exact offset projection."""

    def extract(self, text: str) -> list[EntityAnnotation]:
        return []

