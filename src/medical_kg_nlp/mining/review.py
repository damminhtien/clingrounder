"""Deterministic JSONL exchange format for local or external review backends."""

from __future__ import annotations

import json
from collections.abc import Sequence

from medical_kg_nlp.mining.records import (
    AnnotationLayer,
    AnnotationProposal,
    ConceptLink,
    MinedDocument,
    ReviewStatus,
)

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
                proposal = _proposal_from_dict(raw)
                if proposal.annotation_id in seen:
                    raise ValueError(f"Duplicate reviewed annotation {proposal.annotation_id!r}")
                seen.add(proposal.annotation_id)
                proposals.append(proposal)
        return tuple(proposals)


def _proposal_from_dict(raw: object) -> AnnotationProposal:
    if not isinstance(raw, dict):
        raise ValueError("Review proposal must be an object")
    span = raw.get("span")
    if not isinstance(span, list) or len(span) != 2:
        raise ValueError("Review proposal span must contain start and end")
    concepts = raw.get("concepts", [])
    if not isinstance(concepts, list):
        raise ValueError("Review proposal concepts must be a list")
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Review proposal metadata must be an object")
    return AnnotationProposal(
        annotation_id=str(raw["annotation_id"]),
        document_id=str(raw["document_id"]),
        span=(int(span[0]), int(span[1])),
        text=str(raw["text"]),
        entity_type=str(raw["entity_type"]),
        assertions=tuple(str(value) for value in raw.get("assertions", [])),
        concepts=tuple(
            ConceptLink(
                code_system=str(value["code_system"]),
                code=str(value["code"]),
                terminology_version=str(value["terminology_version"]),
                confidence=float(value.get("confidence", 1.0)),
            )
            for value in concepts
            if isinstance(value, dict)
        ),
        confidence=float(raw["confidence"]),
        layer=AnnotationLayer(str(raw["layer"])),
        label_source=str(raw["label_source"]),
        labeler_id=str(raw["labeler_id"]),
        review_status=ReviewStatus(str(raw["review_status"])),
        source_label=(None if raw.get("source_label") is None else str(raw["source_label"])),
        model_revision=(
            None if raw.get("model_revision") is None else str(raw["model_revision"])
        ),
        prompt_hash=None if raw.get("prompt_hash") is None else str(raw["prompt_hash"]),
        metadata={str(key): str(value) for key, value in metadata.items()},
    )
