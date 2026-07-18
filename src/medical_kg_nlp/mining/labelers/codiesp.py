"""Import source-human CodiEsp-X spans and CIE10 codes as silver proposals."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from medical_kg_nlp.mining.formats.codiesp import (
    CodiEspDocumentBundle,
    CodiEspSpanAnnotation,
    read_codiesp_archive,
)
from medical_kg_nlp.mining.io import load_source_artifacts
from medical_kg_nlp.mining.ports import ArtifactStorePort
from medical_kg_nlp.mining.records import (
    AnnotationLayer,
    AnnotationProposal,
    ConceptLink,
    MinedDocument,
    ReviewStatus,
    SourceArtifact,
)
from medical_kg_nlp.mining.runner import artifact_store_from_uri

__all__ = [
    "CodiEspArchiveLabelerAdapter",
    "CodiEspLabelMapping",
    "create_codiesp_archive_labeler",
]


@dataclass(frozen=True)
class CodiEspLabelMapping:
    """Explicit source label mapping without inferring CodiEsp semantics."""

    entity_type: str
    code_system: str

    def __post_init__(self) -> None:
        if not self.entity_type.strip() or not self.code_system.strip():
            raise ValueError("CodiEsp label mappings must be non-empty")


@dataclass(frozen=True)
class _CodiEspIndexes:
    documents: dict[tuple[str, str], CodiEspDocumentBundle]
    annotations: dict[tuple[str, str], tuple[CodiEspSpanAnnotation, ...]]


class CodiEspArchiveLabelerAdapter:
    """Project CodiEsp-X source labels onto immutable parsed Spanish text."""

    max_artifact_bytes = 512 * 1024 * 1024

    def __init__(
        self,
        *,
        artifacts: Sequence[SourceArtifact],
        store: ArtifactStorePort,
        label_map: Mapping[str, CodiEspLabelMapping],
        labeler_id: str,
        terminology_version: str,
        layer: AnnotationLayer = AnnotationLayer.SILVER,
        review_status: ReviewStatus = ReviewStatus.PROPOSED,
    ) -> None:
        if not labeler_id.strip() or not terminology_version.strip():
            raise ValueError("CodiEsp labeler provenance must be non-empty")
        if not label_map or any(not key.strip() for key in label_map):
            raise ValueError("CodiEsp label_map must contain non-empty source labels")
        self.artifacts = {artifact.artifact_id: artifact for artifact in artifacts}
        if len(self.artifacts) != len(artifacts):
            raise ValueError("CodiEsp artifact manifest contains duplicate artifact IDs")
        self.store = store
        self.label_map = dict(label_map)
        self.labeler_id = labeler_id
        self.terminology_version = terminology_version
        self.layer = layer
        self.review_status = review_status
        self._index_cache: dict[str, _CodiEspIndexes] = {}

    def propose(
        self,
        documents: Sequence[MinedDocument],
    ) -> Iterable[AnnotationProposal]:
        """Emit stable source proposals and quarantine source-offset discrepancies."""

        for document in sorted(documents, key=lambda item: item.document_id):
            split = document.metadata.get("corpus_split")
            case_id = document.metadata.get("codiesp_case_id")
            if not split or not case_id:
                raise ValueError(
                    f"CodiEsp document {document.document_id!r} lacks source identity metadata"
                )
            indexes = self._indexes(document.source_artifact_id)
            source_document = indexes.documents.get((split, case_id))
            if source_document is None:
                raise ValueError(f"Unknown CodiEsp source case {split}/{case_id}")
            if source_document.text != document.text:
                raise ValueError(
                    f"CodiEsp source text changed after parsing for {document.document_id!r}"
                )
            for source_annotation in indexes.annotations.get((split, case_id), ()):
                mapping = self.label_map.get(source_annotation.source_label)
                if mapping is None:
                    raise ValueError(
                        f"Unmapped CodiEsp source label {source_annotation.source_label!r}"
                    )
                start, end = source_annotation.envelope
                issues = list(source_annotation.segment_issues)
                if not source_annotation.source_text_matches:
                    issues.append("source_text_mismatch")
                proposal = AnnotationProposal(
                    annotation_id=_annotation_id(
                        document.document_id,
                        source_annotation.annotation_member,
                        source_annotation.row_number,
                    ),
                    document_id=document.document_id,
                    span=(start, end),
                    # INVARIANT: discontinuous labels use the raw envelope. Source
                    # segments remain serialized for segment-aware model adapters.
                    text=document.text[start:end],
                    entity_type=mapping.entity_type,
                    assertions=(),
                    concepts=(
                        ConceptLink(
                            code_system=mapping.code_system,
                            code=source_annotation.source_code.upper(),
                            terminology_version=self.terminology_version,
                        ),
                    ),
                    confidence=1.0,
                    layer=self.layer,
                    label_source="source_human_annotation",
                    labeler_id=self.labeler_id,
                    review_status=(ReviewStatus.NEEDS_REVIEW if issues else self.review_status),
                    source_label=source_annotation.source_label,
                    metadata={
                        "archive_annotation_member": (source_annotation.annotation_member),
                        "codiesp_segments": _compact_json(source_annotation.segments),
                        "codiesp_raw_segments": _compact_json(source_annotation.raw_segments),
                        "codiesp_row_number": str(source_annotation.row_number),
                        "discontinuous": str(len(source_annotation.segments) > 1).lower(),
                        "import_issues": _compact_json(issues),
                        "source_annotated_text": source_annotation.annotated_text,
                        "source_code": source_annotation.source_code,
                        "source_text_match": str(source_annotation.source_text_matches).lower(),
                    },
                )
                proposal.validate_offsets(document)
                yield proposal

    def _indexes(self, artifact_id: str) -> _CodiEspIndexes:
        cached = self._index_cache.get(artifact_id)
        if cached is not None:
            return cached
        artifact = self.artifacts.get(artifact_id)
        if artifact is None:
            raise ValueError(f"Unknown CodiEsp source artifact {artifact_id!r}")
        if artifact.object.byte_size > self.max_artifact_bytes:
            raise ValueError(
                f"CodiEsp artifact exceeds labeler limit of {self.max_artifact_bytes} bytes"
            )
        with self.store.open(artifact.object.sha256) as stream:
            payload = stream.read(self.max_artifact_bytes + 1)
        if len(payload) > self.max_artifact_bytes:
            raise ValueError(
                f"CodiEsp artifact exceeds labeler limit of {self.max_artifact_bytes} bytes"
            )
        bundle = read_codiesp_archive(payload)
        annotations: dict[tuple[str, str], list[CodiEspSpanAnnotation]] = defaultdict(list)
        for annotation in bundle.annotations:
            annotations[(annotation.split, annotation.case_id)].append(annotation)
        # SCALING: archive parsing and the O(n) annotation grouping happen once per
        # artifact, avoiding a 3,751-document by 18,435-row scan during labeling.
        indexes = _CodiEspIndexes(
            documents={
                (document.split, document.case_id): document for document in bundle.documents
            },
            annotations={key: tuple(values) for key, values in annotations.items()},
        )
        self._index_cache[artifact_id] = indexes
        return indexes


def create_codiesp_archive_labeler(
    config: Mapping[str, Any],
) -> CodiEspArchiveLabelerAdapter:
    """Build a CodiEsp source labeler from a CLI plugin configuration mapping."""

    raw_label_map = config.get("label_map")
    if not isinstance(raw_label_map, Mapping):
        raise ValueError("CodiEsp labeler config requires a label_map mapping")
    label_map = {}
    for source_label, raw_mapping in raw_label_map.items():
        if not isinstance(raw_mapping, Mapping):
            raise ValueError(f"CodiEsp label mapping {source_label!r} must be an object")
        label_map[str(source_label)] = CodiEspLabelMapping(
            entity_type=_required_string(raw_mapping, "entity_type"),
            code_system=_required_string(raw_mapping, "code_system"),
        )
    return CodiEspArchiveLabelerAdapter(
        artifacts=load_source_artifacts(_required_string(config, "artifacts")),
        store=artifact_store_from_uri(_required_string(config, "store")),
        label_map=label_map,
        labeler_id=_required_string(config, "labeler_id"),
        terminology_version=_required_string(config, "terminology_version"),
        layer=AnnotationLayer(str(config.get("layer", AnnotationLayer.SILVER.value))),
        review_status=ReviewStatus(str(config.get("review_status", ReviewStatus.PROPOSED.value))),
    )


def _required_string(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"CodiEsp labeler config requires non-empty {key!r}")
    return value


def _annotation_id(document_id: str, member: str, row_number: int) -> str:
    identity = f"{document_id}\0{member}\0{row_number}".encode("utf-8")
    return f"codiesp:{hashlib.sha256(identity).hexdigest()[:24]}"


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
