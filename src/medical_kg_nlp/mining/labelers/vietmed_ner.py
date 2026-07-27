"""Import VietMed-NER BIO labels from parsed transcript metadata."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from medical_kg_nlp.mining.records import (
    AnnotationLayer,
    AnnotationProposal,
    MinedDocument,
    ReviewStatus,
)

__all__ = [
    "VietMedNerSourceLabelerAdapter",
    "create_vietmed_ner_source_labeler",
    "decode_vietmed_bio_spans",
]


@dataclass(frozen=True, slots=True)
class _DecodedSpan:
    span: tuple[int, int]
    source_label: str


class VietMedNerSourceLabelerAdapter:
    """Emit source-human silver labels while retaining the original 18-type taxonomy."""

    def __init__(
        self,
        *,
        label_map: Mapping[str, str],
        labeler_id: str,
    ) -> None:
        if not labeler_id.strip():
            raise ValueError("VietMed-NER labeler_id must be non-empty")
        if not label_map or any(
            not source.strip() or not target.strip()
            for source, target in label_map.items()
        ):
            raise ValueError("VietMed-NER label_map must contain non-empty values")
        self.label_map = dict(label_map)
        self.labeler_id = labeler_id

    def propose(
        self,
        documents: Sequence[MinedDocument],
    ) -> Iterable[AnnotationProposal]:
        """Decode BIO metadata and validate every envelope against the raw transcript."""

        for document in sorted(documents, key=lambda item: item.document_id):
            offsets = _load_offsets(document)
            labels = _load_labels(document)
            if len(offsets) != len(labels):
                raise ValueError(
                    f"VietMed-NER metadata length mismatch for {document.document_id}"
                )
            for index, decoded in enumerate(
                decode_vietmed_bio_spans(offsets, labels),
                start=1,
            ):
                entity_type = self.label_map.get(decoded.source_label)
                if entity_type is None:
                    raise ValueError(
                        f"Unmapped VietMed-NER source label {decoded.source_label!r}"
                    )
                start, end = decoded.span
                proposal = AnnotationProposal(
                    annotation_id=_annotation_id(
                        document.document_id,
                        index=index,
                        source_label=decoded.source_label,
                        span=decoded.span,
                    ),
                    document_id=document.document_id,
                    span=decoded.span,
                    text=document.text[start:end],
                    entity_type=entity_type,
                    assertions=(),
                    concepts=(),
                    confidence=1.0,
                    layer=AnnotationLayer.SILVER,
                    label_source="source_human_annotation",
                    labeler_id=self.labeler_id,
                    review_status=ReviewStatus.PROPOSED,
                    source_label=decoded.source_label,
                    metadata={
                        "source_taxonomy": "vietmed_ner_18_type",
                        "source_split": document.metadata.get("source_split", ""),
                    },
                )
                proposal.validate_offsets(document)
                yield proposal


def decode_vietmed_bio_spans(
    offsets: Sequence[tuple[int, int]],
    labels: Sequence[str],
) -> tuple[_DecodedSpan, ...]:
    """Decode source BIO labels, treating an orphan ``I`` as a new source span."""

    output: list[_DecodedSpan] = []
    active_label: str | None = None
    active_start = 0
    active_end = 0

    def flush() -> None:
        nonlocal active_label, active_start, active_end
        if active_label is not None:
            output.append(
                _DecodedSpan(
                    span=(active_start, active_end),
                    source_label=active_label,
                )
            )
        active_label = None

    for offset, raw_label in zip(offsets, labels, strict=True):
        start, end = offset
        if start < 0 or end <= start:
            raise ValueError(f"Invalid VietMed-NER token offset: {offset}")
        prefix, source_label = _decode_source_label(raw_label)
        if source_label is None:
            flush()
            continue
        continues = prefix == "I" and active_label == source_label
        if not continues:
            flush()
            active_label = source_label
            active_start = start
        active_end = end
    flush()
    return tuple(output)


def create_vietmed_ner_source_labeler(
    config: Mapping[str, Any],
) -> VietMedNerSourceLabelerAdapter:
    """Build the source labeler used by the generic mining proposal command."""

    raw_label_map = config.get("label_map")
    if not isinstance(raw_label_map, Mapping):
        raise ValueError("VietMed-NER labeler config requires label_map")
    labeler_id = config.get("labeler_id")
    if not isinstance(labeler_id, str) or not labeler_id.strip():
        raise ValueError("VietMed-NER labeler config requires labeler_id")
    return VietMedNerSourceLabelerAdapter(
        label_map={
            str(source): str(target)
            for source, target in raw_label_map.items()
        },
        labeler_id=labeler_id,
    )


def _decode_source_label(raw: str) -> tuple[str, str | None]:
    normalized = raw.strip()
    if normalized in {"", "0", "O"}:
        return "O", None
    prefix, separator, source_label = normalized.partition("-")
    if not separator or prefix not in {"B", "I"} or not source_label:
        raise ValueError(f"Invalid VietMed-NER BIO label: {raw!r}")
    return prefix, source_label


def _load_offsets(document: MinedDocument) -> tuple[tuple[int, int], ...]:
    raw = json.loads(document.metadata.get("token_offsets", "null"))
    if not isinstance(raw, list):
        raise ValueError(f"Missing token_offsets for {document.document_id}")
    offsets: list[tuple[int, int]] = []
    for value in raw:
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"Invalid token_offsets for {document.document_id}")
        offsets.append((int(value[0]), int(value[1])))
    return tuple(offsets)


def _load_labels(document: MinedDocument) -> tuple[str, ...]:
    raw = json.loads(document.metadata.get("source_bio_labels", "null"))
    if not isinstance(raw, list):
        raise ValueError(f"Missing source_bio_labels for {document.document_id}")
    return tuple(str(value) for value in raw)


def _annotation_id(
    document_id: str,
    *,
    index: int,
    source_label: str,
    span: tuple[int, int],
) -> str:
    identity = (
        f"{document_id}\0{index}\0{source_label}\0{span[0]}\0{span[1]}"
    ).encode("utf-8")
    return f"vietmed-ner:{hashlib.sha256(identity).hexdigest()[:24]}"
