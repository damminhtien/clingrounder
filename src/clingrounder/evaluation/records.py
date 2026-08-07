"""Task-neutral records consumed by reusable evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["EvaluationDocument", "EvaluationEntity", "EvaluationRelation"]


@dataclass(frozen=True)
class EvaluationEntity:
    """A span, type, assertions, and candidate identifiers without task schema names."""

    entity_id: str
    span: tuple[int, int]
    text: str
    entity_type: str
    assertions: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()

    def validate_offsets(self, source_text: str) -> None:
        """Reject entities that do not point to their exact source-text slice."""

        start, end = self.span
        if not 0 <= start <= end <= len(source_text):
            raise ValueError(f"Invalid evaluation span {self.span}")
        if source_text[start:end] != self.text:
            raise ValueError(
                f"Evaluation offset mismatch: {source_text[start:end]!r} != {self.text!r}"
            )


@dataclass(frozen=True)
class EvaluationRelation:
    """A typed relation whose endpoints reference neutral entity IDs."""

    relation_id: str
    head: str
    tail: str
    relation_type: str
    evidence_span: tuple[int, int] | None = None


@dataclass(frozen=True)
class EvaluationDocument:
    """All neutral annotations associated with one source document."""

    document_id: str
    entities: tuple[EvaluationEntity, ...]
    relations: tuple[EvaluationRelation, ...] = ()
    source_text: str | None = None

    def validate(self) -> None:
        """Validate offsets and relation endpoint ownership when source text is available."""

        entity_ids = {entity.entity_id for entity in self.entities}
        if len(entity_ids) != len(self.entities):
            raise ValueError(f"Duplicate evaluation entity IDs in {self.document_id}")
        if self.source_text is not None:
            for entity in self.entities:
                entity.validate_offsets(self.source_text)
        for relation in self.relations:
            if relation.head not in entity_ids or relation.tail not in entity_ids:
                raise ValueError(
                    f"Invalid relation endpoints in {self.document_id}: {relation.relation_id}"
                )
