"""Attach raw-offset medication structure to entities produced by model adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from medical_kg_nlp.ner.medication_list_parser import MedicationListParser
from medical_kg_nlp.ner.medication_mention_parser import MedicationMentionParser
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.types import EntityType

if TYPE_CHECKING:
    from medical_kg_nlp.pipeline.ports import EntityExtractorPort

__all__ = ["MedicationMentionEntityExtractorAdapter"]


@dataclass(frozen=True)
class MedicationMentionEntityExtractorAdapter:
    """Decorate model drug proposals with validated full medication mentions.

    This adapter adds structure only. The underlying drug entity remains the recognition span,
    while Phase 1 export and linking may use ``medication_mention.full_span``. It does not create
    indication entities or use the BTC sample lexicon as runtime memory.
    """

    extractor: EntityExtractorPort
    mention_parser: MedicationMentionParser = field(default_factory=MedicationMentionParser)
    list_parser: MedicationListParser = field(default_factory=MedicationListParser)

    def extract(self, source_text: str) -> list[EntityAnnotation]:
        entities = self.extractor.extract(source_text)
        for entity in entities:
            entity.validate_offsets(source_text)
            if entity.type is EntityType.DRUG and entity.medication_mention is None:
                entity.medication_mention = self.mention_parser.parse(
                    source_text,
                    entity.span,
                )
        entities = self.list_parser.adjudicate(source_text, entities)
        for entity in entities:
            entity.validate_offsets(source_text)
            if entity.type is EntityType.DRUG and entity.medication_mention is not None:
                # INVARIANT: full SIG metadata must remain in the same raw coordinate system as
                # the model's original drug span.
                entity.medication_mention.validate_offsets(source_text, entity.span)
        return entities

    def close(self) -> None:
        """Close the wrapped extractor when it owns a model runtime."""

        close = getattr(self.extractor, "close", None)
        if callable(close):
            close()
