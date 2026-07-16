"""Adapters that isolate task schemas from reusable evaluation records."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, TypeVar

from medical_kg_nlp.evaluation.records import EvaluationDocument

__all__ = ["EvaluationAdapter", "adapt_evaluation_records"]

RecordT = TypeVar("RecordT", contravariant=True)


class EvaluationAdapter(Protocol[RecordT]):
    """Convert one task-owned record into a neutral evaluation document."""

    def adapt(self, record: RecordT) -> EvaluationDocument: ...


def adapt_evaluation_records(
    records: Iterable[RecordT],
    adapter: EvaluationAdapter[RecordT],
) -> list[EvaluationDocument]:
    """Adapt and validate records before they enter metric code."""

    documents: list[EvaluationDocument] = []
    seen: set[str] = set()
    for record in records:
        document = adapter.adapt(record)
        if document.document_id in seen:
            raise ValueError(f"Duplicate evaluation document ID: {document.document_id}")
        document.validate()
        seen.add(document.document_id)
        documents.append(document)
    return documents
