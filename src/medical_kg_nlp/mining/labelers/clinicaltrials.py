"""Project ClinicalTrials.gov source fields into silver spans and neutral relations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from medical_kg_nlp.mining.records import (
    AnnotationLayer,
    AnnotationProposal,
    MinedDocument,
    RelationProposal,
    ReviewStatus,
)

__all__ = [
    "ClinicalTrialsStructuredLabelerAdapter",
    "ClinicalTrialsStructuredRelationLabelerAdapter",
    "create_clinicaltrials_structured_labeler",
    "create_clinicaltrials_structured_relation_labeler",
]


class ClinicalTrialsStructuredLabelerAdapter:
    """Emit only fields projected by the API v2 parser."""

    def __init__(self, *, labeler_id: str) -> None:
        if not labeler_id.strip():
            raise ValueError("ClinicalTrials labeler_id must be non-empty")
        self.labeler_id = labeler_id

    def propose(
        self, documents: Sequence[MinedDocument]
    ) -> Iterable[AnnotationProposal]:
        for document in sorted(documents, key=lambda item: item.document_id):
            if document.note_type != "clinical_trial":
                continue
            for index, field in enumerate(_load_fields(document)):
                start, end = _span(field.get("span"), document.document_id)
                proposal = AnnotationProposal(
                    annotation_id=_annotation_id(document.document_id, index, field),
                    document_id=document.document_id,
                    span=(start, end),
                    text=_required_string(field, "text"),
                    entity_type=_required_string(field, "entity_type"),
                    assertions=(),
                    concepts=(),
                    confidence=1.0,
                    layer=AnnotationLayer.SILVER,
                    label_source="source_structured_annotation",
                    labeler_id=self.labeler_id,
                    review_status=ReviewStatus.PROPOSED,
                    source_label=_required_string(field, "source_label"),
                    metadata={
                        "clinicaltrials_nct_id": _required_metadata(
                            document, "clinicaltrials_nct_id"
                        ),
                        "field_role": _required_string(field, "role"),
                        "relation_group": _required_string(field, "group_id"),
                    },
                )
                proposal.validate_offsets(document)
                yield proposal


class ClinicalTrialsStructuredRelationLabelerAdapter:
    """Relate registered interventions to conditions without claiming treatment efficacy."""

    def __init__(self, *, labeler_id: str) -> None:
        if not labeler_id.strip():
            raise ValueError("ClinicalTrials relation labeler_id must be non-empty")
        self.labeler_id = labeler_id

    def propose(
        self,
        documents: Sequence[MinedDocument],
        annotations: Sequence[AnnotationProposal],
    ) -> Iterable[RelationProposal]:
        by_document: dict[str, list[AnnotationProposal]] = {}
        for annotation in annotations:
            by_document.setdefault(annotation.document_id, []).append(annotation)
        for document in sorted(documents, key=lambda item: item.document_id):
            values = by_document.get(document.document_id, ())
            conditions = sorted(
                (value for value in values if value.metadata.get("field_role") == "condition"),
                key=lambda item: item.annotation_id,
            )
            interventions = sorted(
                (
                    value
                    for value in values
                    if value.metadata.get("field_role") == "intervention"
                ),
                key=lambda item: item.annotation_id,
            )
            for condition in conditions:
                for intervention in interventions:
                    relation_type = "STUDIES_INTERVENTION"
                    identity = (
                        f"{document.document_id}\0{condition.annotation_id}\0"
                        f"{intervention.annotation_id}\0{relation_type}"
                    )
                    yield RelationProposal(
                        relation_id=(
                            "clinicaltrials:"
                            + hashlib.sha256(identity.encode()).hexdigest()[:24]
                        ),
                        document_id=document.document_id,
                        head_annotation_id=condition.annotation_id,
                        tail_annotation_id=intervention.annotation_id,
                        relation_type=relation_type,
                        confidence=1.0,
                        layer=AnnotationLayer.SILVER,
                        label_source="source_structured_relation",
                        evidence_span=(
                            min(condition.span[0], intervention.span[0]),
                            max(condition.span[1], intervention.span[1]),
                        ),
                        labeler_id=self.labeler_id,
                        review_status=ReviewStatus.PROPOSED,
                        metadata={
                            "clinicaltrials_nct_id": _required_metadata(
                                document, "clinicaltrials_nct_id"
                            ),
                            "semantics": "registered intervention studied for condition",
                        },
                    )


def create_clinicaltrials_structured_labeler(
    config: Mapping[str, Any],
) -> ClinicalTrialsStructuredLabelerAdapter:
    """Build the source-field labeler from task-neutral plugin configuration."""

    return ClinicalTrialsStructuredLabelerAdapter(
        labeler_id=_required_string(config, "labeler_id")
    )


def create_clinicaltrials_structured_relation_labeler(
    config: Mapping[str, Any],
) -> ClinicalTrialsStructuredRelationLabelerAdapter:
    """Build the neutral condition-intervention relation labeler."""

    return ClinicalTrialsStructuredRelationLabelerAdapter(
        labeler_id=_required_string(config, "labeler_id")
    )


def _load_fields(document: MinedDocument) -> tuple[Mapping[str, Any], ...]:
    try:
        payload = json.loads(_required_metadata(document, "clinicaltrials_fields"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid clinicaltrials_fields for {document.document_id!r}"
        ) from error
    if not isinstance(payload, list) or any(not isinstance(item, Mapping) for item in payload):
        raise ValueError(
            f"clinicaltrials_fields for {document.document_id!r} must be an object list"
        )
    return tuple(payload)


def _span(value: Any, document_id: str) -> tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ValueError(f"ClinicalTrials field in {document_id!r} has invalid span")
    return int(value[0]), int(value[1])


def _annotation_id(
    document_id: str, index: int, field: Mapping[str, Any]
) -> str:
    identity = (
        f"{document_id}\0{index}\0{field.get('span')}\0{field.get('source_label')}"
    )
    return f"clinicaltrials:{hashlib.sha256(identity.encode()).hexdigest()[:24]}"


def _required_string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"ClinicalTrials field requires non-empty {key!r}")
    return result


def _required_metadata(document: MinedDocument, key: str) -> str:
    value = document.metadata.get(key, "")
    if not value.strip():
        raise ValueError(f"ClinicalTrials document {document.document_id!r} lacks {key!r}")
    return value
