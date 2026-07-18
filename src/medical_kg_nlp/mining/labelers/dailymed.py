"""Project source-structured DailyMed SPL fields into silver annotations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from medical_kg_nlp.mining.records import (
    AnnotationLayer,
    AnnotationProposal,
    ConceptLink,
    MinedDocument,
    RelationProposal,
    ReviewStatus,
)

__all__ = [
    "DailyMedStructuredLabelerAdapter",
    "DailyMedStructuredRelationLabelerAdapter",
    "create_dailymed_structured_labeler",
    "create_dailymed_structured_relation_labeler",
]


class DailyMedStructuredLabelerAdapter:
    """Read parser-projected SPL fields without weak-matching narrative text."""

    def __init__(
        self,
        *,
        labeler_id: str,
        layer: AnnotationLayer = AnnotationLayer.SILVER,
        review_status: ReviewStatus = ReviewStatus.PROPOSED,
    ) -> None:
        if not labeler_id.strip():
            raise ValueError("DailyMed labeler_id must be non-empty")
        self.labeler_id = labeler_id
        self.layer = layer
        self.review_status = review_status

    def propose(
        self,
        documents: Sequence[MinedDocument],
    ) -> Iterable[AnnotationProposal]:
        """Emit annotations only for deterministic structured medication records."""

        for document in sorted(documents, key=lambda item: item.document_id):
            if document.note_type != "structured_medication_record":
                continue
            fields = _load_fields(document)
            terminology_version = _required_metadata(document, "dailymed_source_version")
            for index, field in enumerate(fields):
                start, end = _span(field.get("span"), document_id=document.document_id)
                text = _required_string(field, "text")
                code_system = str(field.get("code_system", "")).strip()
                code = str(field.get("code", "")).strip()
                if bool(code_system) != bool(code):
                    raise ValueError(
                        f"DailyMed field {index} must define both code and code_system"
                    )
                proposal = AnnotationProposal(
                    annotation_id=_annotation_id(document.document_id, index, field),
                    document_id=document.document_id,
                    span=(start, end),
                    text=text,
                    entity_type=_required_string(field, "entity_type"),
                    assertions=(),
                    concepts=(
                        ()
                        if not code
                        else (
                            ConceptLink(
                                code_system=code_system,
                                code=code,
                                terminology_version=terminology_version,
                            ),
                        )
                    ),
                    confidence=1.0,
                    layer=self.layer,
                    label_source="source_structured_annotation",
                    labeler_id=self.labeler_id,
                    review_status=self.review_status,
                    source_label=_required_string(field, "source_label"),
                    metadata={
                        "dailymed_set_id": _required_metadata(
                            document, "dailymed_set_id"
                        ),
                        "field_role": _required_string(field, "role"),
                        "relation_group": _required_string(field, "group_id"),
                        "spl_product_index": _required_metadata(
                            document, "spl_product_index"
                        ),
                    },
                )
                # INVARIANT: source-structured labels still target immutable rendered text.
                proposal.validate_offsets(document)
                yield proposal


class DailyMedStructuredRelationLabelerAdapter:
    """Derive medication composition and attribute relations from SPL roles."""

    def __init__(
        self,
        *,
        labeler_id: str,
        layer: AnnotationLayer = AnnotationLayer.SILVER,
        review_status: ReviewStatus = ReviewStatus.PROPOSED,
    ) -> None:
        if not labeler_id.strip():
            raise ValueError("DailyMed relation labeler_id must be non-empty")
        self.labeler_id = labeler_id
        self.layer = layer
        self.review_status = review_status

    def propose(
        self,
        documents: Sequence[MinedDocument],
        annotations: Sequence[AnnotationProposal],
    ) -> Iterable[RelationProposal]:
        """Connect product/ingredient heads to their source-structured tails."""

        annotations_by_document: dict[str, list[AnnotationProposal]] = {}
        for annotation in annotations:
            annotations_by_document.setdefault(annotation.document_id, []).append(annotation)
        for document in sorted(documents, key=lambda item: item.document_id):
            if document.note_type != "structured_medication_record":
                continue
            values = annotations_by_document.get(document.document_id, [])
            products = [
                value for value in values if value.metadata.get("field_role") == "product"
            ]
            if len(products) != 1:
                raise ValueError(
                    f"DailyMed record {document.document_id!r} requires exactly one product"
                )
            product = products[0]
            ingredients = {
                value.metadata.get("relation_group", ""): value
                for value in values
                if value.metadata.get("field_role") == "active_ingredient"
            }
            for tail in sorted(values, key=lambda item: item.annotation_id):
                role = tail.metadata.get("field_role", "")
                relation_type = _RELATION_BY_ROLE.get(role)
                if relation_type is None:
                    continue
                head = product
                if role == "strength":
                    group_id = tail.metadata.get("relation_group", "")
                    ingredient_head = ingredients.get(group_id)
                    if ingredient_head is None:
                        raise ValueError(
                            f"DailyMed strength {tail.annotation_id!r} has no ingredient head"
                        )
                    head = ingredient_head
                start = min(head.span[0], tail.span[0])
                end = max(head.span[1], tail.span[1])
                yield RelationProposal(
                    relation_id=_relation_id(
                        document.document_id,
                        head.annotation_id,
                        tail.annotation_id,
                        relation_type,
                    ),
                    document_id=document.document_id,
                    head_annotation_id=head.annotation_id,
                    tail_annotation_id=tail.annotation_id,
                    relation_type=relation_type,
                    confidence=1.0,
                    layer=self.layer,
                    label_source="source_structured_relation",
                    evidence_span=(start, end),
                    labeler_id=self.labeler_id,
                    review_status=self.review_status,
                    metadata={
                        "dailymed_set_id": _required_metadata(
                            document, "dailymed_set_id"
                        ),
                        "spl_product_index": _required_metadata(
                            document, "spl_product_index"
                        ),
                    },
                )


def create_dailymed_structured_labeler(
    config: Mapping[str, Any],
) -> DailyMedStructuredLabelerAdapter:
    """Build the DailyMed labeler from task-neutral CLI plugin configuration."""

    return DailyMedStructuredLabelerAdapter(
        labeler_id=_required_string(config, "labeler_id"),
        layer=AnnotationLayer(str(config.get("layer", AnnotationLayer.SILVER.value))),
        review_status=ReviewStatus(
            str(config.get("review_status", ReviewStatus.PROPOSED.value))
        ),
    )


def create_dailymed_structured_relation_labeler(
    config: Mapping[str, Any],
) -> DailyMedStructuredRelationLabelerAdapter:
    """Build the DailyMed relation labeler for the generic relation CLI."""

    return DailyMedStructuredRelationLabelerAdapter(
        labeler_id=_required_string(config, "labeler_id"),
        layer=AnnotationLayer(str(config.get("layer", AnnotationLayer.SILVER.value))),
        review_status=ReviewStatus(
            str(config.get("review_status", ReviewStatus.PROPOSED.value))
        ),
    )


def _load_fields(document: MinedDocument) -> tuple[Mapping[str, Any], ...]:
    raw = _required_metadata(document, "spl_fields")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid spl_fields for {document.document_id!r}") from error
    if not isinstance(payload, list):
        raise ValueError(f"spl_fields for {document.document_id!r} must be a list")
    fields: list[Mapping[str, Any]] = []
    for field in payload:
        if not isinstance(field, Mapping):
            raise ValueError(f"spl_fields for {document.document_id!r} contains a non-object")
        fields.append(field)
    return tuple(fields)


def _span(value: Any, *, document_id: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"DailyMed field in {document_id!r} has an invalid span")
    return value[0], value[1]


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"DailyMed labeler requires non-empty {key!r}")
    return value.strip()


def _required_metadata(document: MinedDocument, key: str) -> str:
    value = document.metadata.get(key, "").strip()
    if not value:
        raise ValueError(f"DailyMed document {document.document_id!r} lacks {key!r}")
    return value


def _annotation_id(
    document_id: str,
    index: int,
    field: Mapping[str, Any],
) -> str:
    identity = (
        f"{document_id}\0{index}\0{field.get('source_label', '')}\0"
        f"{field.get('role', '')}"
    ).encode("utf-8")
    return f"dailymed:{hashlib.sha256(identity).hexdigest()[:24]}"


def _relation_id(
    document_id: str,
    head_id: str,
    tail_id: str,
    relation_type: str,
) -> str:
    identity = f"{document_id}\0{head_id}\0{tail_id}\0{relation_type}".encode("utf-8")
    return f"dailymed-rel:{hashlib.sha256(identity).hexdigest()[:24]}"


_RELATION_BY_ROLE = {
    "generic_name": "HAS_GENERIC_NAME",
    "active_ingredient": "HAS_ACTIVE_INGREDIENT",
    "strength": "HAS_STRENGTH",
    "dosage_form": "HAS_DOSAGE_FORM",
    "route": "HAS_ROUTE",
}
