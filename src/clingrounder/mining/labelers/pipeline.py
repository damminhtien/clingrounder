"""Run the local clinical pipeline as a provenance-bearing mining labeler.

The adapter deliberately emits proposals rather than gold annotations.  This makes it possible
to mine rare cases with the same NER, context, and terminology configuration used by an
application while keeping human review and promotion as separate stages.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol

from clingrounder.mining.records import (
    AnnotationLayer,
    AnnotationProposal,
    ConceptLink,
    MinedDocument,
    ReviewStatus,
)
from clingrounder.pipeline.config_loader import ResolvedPipelineConfig
from clingrounder.pipeline.factory import PipelineFactory
from clingrounder.schema.output import ClinicalPrediction
from clingrounder.schema.types import AssertionStatus, CodeSystem

__all__ = [
    "LocalPipelineProposalLabeler",
    "create_local_pipeline_labeler",
]


class _PipelinePort(Protocol):
    """Minimal runner contract so tests and external local runners stay replaceable."""

    def process_text(
        self,
        document_id: str,
        text: str,
        metadata: dict[str, str] | None = None,
    ) -> ClinicalPrediction: ...


class LocalPipelineProposalLabeler:
    """Convert local pipeline predictions into reviewable mining proposals.

    The labeler never changes source text and never promotes predictions to gold.  Assigned
    codes are copied only from the validated prediction; unlinked candidates remain unlinked in
    the proposal so a later terminology review can decide whether a mapping is justified.
    """

    def __init__(
        self,
        runner: _PipelinePort,
        *,
        labeler_id: str,
        terminology_versions: Mapping[str, str] | None = None,
        layer: AnnotationLayer = AnnotationLayer.BRONZE,
        review_status: ReviewStatus = ReviewStatus.PROPOSED,
        model_revision: str | None = None,
    ) -> None:
        if not labeler_id.strip():
            raise ValueError("Local pipeline labeler_id must be non-empty")
        self.runner = runner
        self.labeler_id = labeler_id
        self.terminology_versions = dict(terminology_versions or {})
        self.layer = layer
        self.review_status = review_status
        self.model_revision = model_revision

    def propose(
        self,
        documents: Sequence[MinedDocument],
    ) -> Iterable[AnnotationProposal]:
        """Run documents in stable order and validate every projected proposal."""

        for document in sorted(documents, key=lambda item: item.document_id):
            prediction = self.runner.process_text(
                document.document_id,
                document.text,
                metadata={"mining_source_artifact_id": document.source_artifact_id},
            )
            if prediction.document_id != document.document_id:
                raise ValueError(
                    f"Pipeline returned document {prediction.document_id!r} for "
                    f"{document.document_id!r}"
                )
            for entity in prediction.entities:
                # INVARIANT: a model/rule adapter cannot bypass the immutable source slice.
                entity.validate_offsets(document.text)
                proposal = self._proposal(document, entity)
                proposal.validate_offsets(document)
                yield proposal

    def _proposal(self, document: MinedDocument, entity: Any) -> AnnotationProposal:
        concepts: tuple[ConceptLink, ...] = ()
        if entity.code is not None and entity.code_system is not CodeSystem.NONE:
            code_system = entity.code_system.value
            concepts = (
                ConceptLink(
                    code_system=code_system,
                    code=str(entity.code),
                    terminology_version=self.terminology_versions.get(
                        code_system,
                        code_system,
                    ),
                ),
            )
        assertions = tuple(
            status.value
            for status in entity.assertion_features.statuses()
            if status not in {AssertionStatus.PRESENT, AssertionStatus.UNKNOWN}
        )
        identity = (
            f"{document.document_id}\0{entity.span}\0{entity.type.value}\0"
            f"{self.labeler_id}"
        )
        metadata = {
            "pipeline_version": str(
                getattr(getattr(self.runner, "components", None), "pipeline_version", "unknown")
            ),
            "source_entity_id": str(entity.id),
            "candidate_count": str(len(entity.candidates)),
            "prediction_code_system": entity.code_system.value,
        }
        return AnnotationProposal(
            annotation_id=f"pipeline:{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}",
            document_id=document.document_id,
            span=entity.span,
            text=entity.text,
            entity_type=entity.type.value,
            assertions=assertions,
            concepts=concepts,
            confidence=max(0.0, min(1.0, float(entity.confidence))),
            layer=self.layer,
            label_source="local_pipeline_prediction",
            labeler_id=self.labeler_id,
            review_status=self.review_status,
            source_label=entity.type.value,
            model_revision=self.model_revision,
            metadata=metadata,
        )


def create_local_pipeline_labeler(
    config: Mapping[str, Any],
) -> LocalPipelineProposalLabeler:
    """Build a local-only labeler from a pinned pipeline YAML configuration."""

    pipeline_config_path = _required_string(config, "pipeline_config")
    resolved_pipeline = ResolvedPipelineConfig.load(pipeline_config_path)
    runner = PipelineFactory.from_config(resolved_pipeline.factory_config)
    versions = config.get("terminology_versions", {})
    if not isinstance(versions, Mapping):
        raise ValueError("terminology_versions must be a mapping")
    model_revision = config.get("model_revision")
    if model_revision is not None and not isinstance(model_revision, str):
        raise ValueError("model_revision must be a string when supplied")
    return LocalPipelineProposalLabeler(
        runner,
        labeler_id=_required_string(config, "labeler_id"),
        terminology_versions={str(key): str(value) for key, value in versions.items()},
        layer=AnnotationLayer(str(config.get("layer", AnnotationLayer.BRONZE.value))),
        review_status=ReviewStatus(
            str(config.get("review_status", ReviewStatus.PROPOSED.value))
        ),
        model_revision=model_revision,
    )


def _required_string(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Local pipeline labeler config requires non-empty {key!r}")
    return value
