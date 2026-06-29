from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from medical_kg_nlp.schema.annotation import EntityAnnotation, RelationAnnotation
from medical_kg_nlp.utils.hashing import sha256_text


@dataclass(frozen=True)
class PredictionMetadata:
    pipeline_version: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> dict[str, Any]:
        return {"pipeline_version": self.pipeline_version, "created_at": self.created_at}


@dataclass
class ClinicalPrediction:
    document_id: str
    text_hash: str
    entities: list[EntityAnnotation]
    relations: list[RelationAnnotation]
    metadata: PredictionMetadata

    @classmethod
    def from_text(
        cls,
        document_id: str,
        text: str,
        entities: list[EntityAnnotation],
        relations: list[RelationAnnotation],
        pipeline_version: str,
    ) -> "ClinicalPrediction":
        return cls(
            document_id=document_id,
            text_hash=sha256_text(text),
            entities=entities,
            relations=relations,
            metadata=PredictionMetadata(pipeline_version=pipeline_version),
        )

    def validate(self, source_text: str) -> None:
        for entity in self.entities:
            entity.validate_offsets(source_text)

    def to_json(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "text_hash": self.text_hash,
            "entities": [entity.to_json() for entity in self.entities],
            "relations": [relation.to_json() for relation in self.relations],
            "metadata": self.metadata.to_json(),
        }
