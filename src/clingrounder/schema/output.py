from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from clingrounder.schema.annotation import EntityAnnotation, RelationAnnotation
from clingrounder.utils.hashing import sha256_text


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

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("Prediction document_id must be non-empty")
        # Intermediate parsed fixtures may omit a hash; ``from_text`` and release validation
        # always provide one before a prediction is published.
        entity_ids = [entity.id for entity in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("Prediction entity IDs must be unique")
        relation_ids = [relation.id for relation in self.relations]
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("Prediction relation IDs must be unique")

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
            concept_ids = [candidate.concept_id for candidate in entity.candidates]
            if len(concept_ids) != len(set(concept_ids)):
                raise ValueError(f"Duplicate candidate concept IDs for entity {entity.id}")
        entity_ids = {entity.id for entity in self.entities}
        for relation in self.relations:
            if relation.head not in entity_ids or relation.tail not in entity_ids:
                raise ValueError(f"Relation {relation.id} references an unknown entity")
            if relation.evidence_span is not None:
                start, end = relation.evidence_span
                if not 0 <= start < end <= len(source_text):
                    raise ValueError(f"Invalid relation evidence span {relation.evidence_span}")
            if relation.evidence is not None and relation.evidence.evidence_span is not None:
                start, end = relation.evidence.evidence_span
                if not 0 <= start < end <= len(source_text):
                    raise ValueError(
                        f"Invalid nested relation evidence span {relation.evidence.evidence_span}"
                    )

    def to_json(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "text_hash": self.text_hash,
            "entities": [entity.to_json() for entity in self.entities],
            "relations": [relation.to_json() for relation in self.relations],
            "metadata": self.metadata.to_json(),
        }
