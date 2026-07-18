"""Deterministic JSONL exchange format for local or external review backends."""

from __future__ import annotations

import json
from collections.abc import Sequence

from medical_kg_nlp.mining.records import (
    AnnotationProposal,
    MinedDocument,
)
from medical_kg_nlp.mining.io import annotation_from_dict

__all__ = ["JsonlReviewBackend"]


class JsonlReviewBackend:
    """Export one document per line and import provenance-preserving decisions."""

    schema_version = "medical-review-queue.v1"

    def export(
        self,
        documents: Sequence[MinedDocument],
        annotations: Sequence[AnnotationProposal],
    ) -> str:
        annotations_by_document: dict[str, list[AnnotationProposal]] = {}
        for annotation in annotations:
            annotations_by_document.setdefault(annotation.document_id, []).append(annotation)
        lines: list[str] = []
        for document in sorted(documents, key=lambda item: item.document_id):
            proposals = sorted(
                annotations_by_document.get(document.document_id, []),
                key=lambda item: (item.span, item.entity_type, item.annotation_id),
            )
            payload = {
                "schema_version": self.schema_version,
                "document": document.to_dict(),
                "proposals": [proposal.to_dict() for proposal in proposals],
            }
            lines.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return "\n".join(lines) + ("\n" if lines else "")

    def import_reviewed(self, payload: str) -> Sequence[AnnotationProposal]:
        proposals: list[AnnotationProposal] = []
        seen: set[str] = set()
        for line_number, line in enumerate(payload.splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != self.schema_version:
                raise ValueError(f"Unsupported review schema on line {line_number}")
            raw_proposals = record.get("proposals")
            if not isinstance(raw_proposals, list):
                raise ValueError(f"Review proposals must be a list on line {line_number}")
            for raw in raw_proposals:
                if not isinstance(raw, dict):
                    raise ValueError("Review proposal must be an object")
                proposal = annotation_from_dict(raw)
                if proposal.annotation_id in seen:
                    raise ValueError(f"Duplicate reviewed annotation {proposal.annotation_id!r}")
                seen.add(proposal.annotation_id)
                proposals.append(proposal)
        return tuple(proposals)
