"""Import manual BRAT labels as provenance-bearing silver proposals."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from clingrounder.mining.formats.brat import (
    BratDocumentBundle,
    parse_brat_text_bound_annotations,
    read_brat_archive,
)
from clingrounder.mining.io import load_source_artifacts
from clingrounder.mining.ports import ArtifactStorePort
from clingrounder.mining.records import (
    AnnotationLayer,
    AnnotationProposal,
    MinedDocument,
    ReviewStatus,
    SourceArtifact,
)
from clingrounder.mining.runner import artifact_store_from_uri

__all__ = ["BratArchiveLabelerAdapter", "create_brat_archive_labeler"]


class BratArchiveLabelerAdapter:
    """Read source-human BRAT labels without treating them as internal gold."""

    max_artifact_bytes = 512 * 1024 * 1024

    def __init__(
        self,
        *,
        artifacts: Sequence[SourceArtifact],
        store: ArtifactStorePort,
        label_map: Mapping[str, str],
        labeler_id: str,
        layer: AnnotationLayer = AnnotationLayer.SILVER,
        review_status: ReviewStatus = ReviewStatus.PROPOSED,
    ) -> None:
        if not labeler_id.strip():
            raise ValueError("BRAT labeler_id must be non-empty")
        if not label_map or any(
            not source.strip() or not target.strip() for source, target in label_map.items()
        ):
            raise ValueError("BRAT label_map must contain non-empty source and target labels")
        self.artifacts = {artifact.artifact_id: artifact for artifact in artifacts}
        if len(self.artifacts) != len(artifacts):
            raise ValueError("BRAT artifact manifest contains duplicate artifact IDs")
        self.store = store
        self.label_map = dict(label_map)
        self.labeler_id = labeler_id
        self.layer = layer
        self.review_status = review_status
        self._bundle_cache: dict[str, dict[str, BratDocumentBundle]] = {}

    def propose(
        self,
        documents: Sequence[MinedDocument],
    ) -> Iterable[AnnotationProposal]:
        """Emit deterministic proposals and validate every envelope against raw text."""

        for document in sorted(documents, key=lambda item: item.document_id):
            annotation_member = document.metadata.get("annotation_member")
            if not annotation_member:
                raise ValueError(
                    f"BRAT document {document.document_id!r} has no annotation_member metadata"
                )
            bundle = self._bundles(document.source_artifact_id).get(annotation_member)
            if bundle is None:
                raise ValueError(
                    f"BRAT annotation member {annotation_member!r} is absent from source archive"
                )
            if bundle.text != document.text:
                raise ValueError(
                    f"BRAT source text changed after parsing for {document.document_id!r}"
                )
            for source_annotation in parse_brat_text_bound_annotations(
                bundle.annotations,
                source_text=document.text,
            ):
                entity_type = self.label_map.get(source_annotation.source_label)
                if entity_type is None:
                    raise ValueError(
                        f"Unmapped BRAT source label {source_annotation.source_label!r}"
                    )
                start, end = source_annotation.envelope
                annotation = AnnotationProposal(
                    annotation_id=_annotation_id(
                        document.document_id,
                        annotation_member,
                        source_annotation.annotation_id,
                    ),
                    document_id=document.document_id,
                    span=(start, end),
                    # INVARIANT: discontinuous source labels use a raw envelope; original
                    # segments remain available in metadata for later structured models.
                    text=document.text[start:end],
                    entity_type=entity_type,
                    assertions=(),
                    concepts=(),
                    confidence=1.0,
                    layer=self.layer,
                    label_source="source_human_annotation",
                    labeler_id=self.labeler_id,
                    review_status=self.review_status,
                    source_label=source_annotation.source_label,
                    metadata={
                        "archive_annotation_member": annotation_member,
                        "brat_annotation_id": source_annotation.annotation_id,
                        "brat_segments": json.dumps(
                            source_annotation.segments,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        "discontinuous": str(len(source_annotation.segments) > 1).lower(),
                        "source_annotated_text": source_annotation.annotated_text,
                    },
                )
                annotation.validate_offsets(document)
                yield annotation

    def _bundles(self, artifact_id: str) -> dict[str, BratDocumentBundle]:
        cached = self._bundle_cache.get(artifact_id)
        if cached is not None:
            return cached
        artifact = self.artifacts.get(artifact_id)
        if artifact is None:
            raise ValueError(f"Unknown BRAT source artifact {artifact_id!r}")
        if artifact.object.byte_size > self.max_artifact_bytes:
            raise ValueError(
                f"BRAT artifact exceeds labeler limit of {self.max_artifact_bytes} bytes"
            )
        with self.store.open(artifact.object.sha256) as stream:
            payload = stream.read(self.max_artifact_bytes + 1)
        if len(payload) > self.max_artifact_bytes:
            raise ValueError(
                f"BRAT artifact exceeds labeler limit of {self.max_artifact_bytes} bytes"
            )
        # SCALING: one parsed archive is reused across CLI batches; only source text and
        # annotations are cached, never model tensors or transformed text.
        bundles = {bundle.annotation_member: bundle for bundle in read_brat_archive(payload)}
        self._bundle_cache[artifact_id] = bundles
        return bundles


def create_brat_archive_labeler(
    config: Mapping[str, Any],
) -> BratArchiveLabelerAdapter:
    """Build a BRAT source labeler from a CLI plugin configuration mapping."""

    artifacts_path = _required_string(config, "artifacts")
    store_uri = _required_string(config, "store")
    labeler_id = _required_string(config, "labeler_id")
    raw_label_map = config.get("label_map")
    if not isinstance(raw_label_map, Mapping):
        raise ValueError("BRAT labeler config requires a label_map mapping")
    label_map = {str(source): str(target) for source, target in raw_label_map.items()}
    layer = AnnotationLayer(str(config.get("layer", AnnotationLayer.SILVER.value)))
    review_status = ReviewStatus(str(config.get("review_status", ReviewStatus.PROPOSED.value)))
    return BratArchiveLabelerAdapter(
        artifacts=load_source_artifacts(artifacts_path),
        store=artifact_store_from_uri(store_uri),
        label_map=label_map,
        labeler_id=labeler_id,
        layer=layer,
        review_status=review_status,
    )


def _required_string(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"BRAT labeler config requires non-empty {key!r}")
    return value


def _annotation_id(document_id: str, member: str, source_id: str) -> str:
    identity = f"{document_id}\0{member}\0{source_id}".encode("utf-8")
    return f"brat:{hashlib.sha256(identity).hexdigest()[:24]}"
