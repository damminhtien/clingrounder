"""Adapt the organizer's flat Phase 1 JSON schema to neutral evaluation records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from medical_kg_nlp.evaluation.records import EvaluationDocument, EvaluationEntity

__all__ = ["Phase1EvaluationAdapter", "Phase1Record"]


@dataclass(frozen=True)
class Phase1Record:
    """One Phase 1 document and its flat entity array."""

    document_id: str
    entities: Sequence[Mapping[str, object]]
    source_text: str | None = None


class Phase1EvaluationAdapter:
    """Convert Vietnamese labels and optional fields without leaking them into metrics."""

    def adapt(self, record: Phase1Record) -> EvaluationDocument:
        entities: list[EvaluationEntity] = []
        for index, row in enumerate(record.entities, start=1):
            position = row.get("position")
            if not isinstance(position, list | tuple) or len(position) != 2:
                raise ValueError(f"Invalid Phase 1 position in {record.document_id}: {position}")
            assertions = _string_sequence(row.get("assertions", ()), "assertions")
            candidates = _string_sequence(row.get("candidates", ()), "candidates")
            entities.append(
                EvaluationEntity(
                    entity_id=f"E{index}",
                    span=(int(position[0]), int(position[1])),
                    text=str(row.get("text", "")),
                    entity_type=str(row.get("type", "")),
                    assertions=assertions,
                    candidates=candidates,
                )
            )
        return EvaluationDocument(
            document_id=record.document_id,
            entities=tuple(entities),
            source_text=record.source_text,
        )


def _string_sequence(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"Phase 1 {field} must be an array")
    return tuple(str(item) for item in value)
