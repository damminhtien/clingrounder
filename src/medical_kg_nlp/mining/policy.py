"""License, privacy, provenance, and annotation quality gates."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass

from medical_kg_nlp.mining.records import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    MinedDocument,
    RedistributionPolicy,
    ReviewStatus,
    SourceArtifact,
)
from medical_kg_nlp.mining.registry import (
    LicenseMode,
    RetentionPolicy,
    SourceDefinition,
    SourceRegistry,
)

__all__ = ["MiningQualityGate", "PolicyDecision", "SourcePolicyGate"]


@dataclass(frozen=True)
class PolicyDecision:
    """One explicit allow/deny result suitable for manifests and audit logs."""

    allowed: bool
    reasons: tuple[str, ...] = ()


class SourcePolicyGate:
    """Apply registry policy before bytes reach hosted or redistributable workflows."""

    def __init__(self, registry: SourceRegistry) -> None:
        self.registry = registry

    def validate_artifact(self, artifact: SourceArtifact) -> PolicyDecision:
        reasons: list[str] = []
        try:
            source = self.registry.by_id(artifact.source_id)
        except KeyError:
            return PolicyDecision(False, (f"unregistered_source:{artifact.source_id}",))
        if source.version_policy.value == "pinned" and artifact.source_version != source.version:
            reasons.append("source_version_mismatch")
        if artifact.access_class is not source.access_class:
            reasons.append("access_class_mismatch")
        if source.license_mode is LicenseMode.FIXED and artifact.license_id != source.license_id:
            reasons.append("license_mismatch")
        if artifact.redistribution is not source.redistribution:
            reasons.append("redistribution_mismatch")
        if artifact.hosted_processing_allowed != source.hosted_processing_allowed:
            reasons.append("hosted_processing_mismatch")
        return PolicyDecision(not reasons, tuple(reasons))

    def hosted_processing(self, document: MinedDocument) -> PolicyDecision:
        reasons: list[str] = []
        if not document.hosted_processing_allowed:
            reasons.append("document_disallows_hosted_processing")
        if document.access_class in {
            AccessClass.CREDENTIALLED,
            AccessClass.DUA,
            AccessClass.LOCAL_PRIVATE,
            AccessClass.QUARANTINE,
        }:
            reasons.append(f"restricted_access:{document.access_class.value}")
        return PolicyDecision(not reasons, tuple(reasons))

    def redistribution(self, documents: Sequence[MinedDocument]) -> PolicyDecision:
        reasons: set[str] = set()
        for document in documents:
            if document.redistribution in {
                RedistributionPolicy.PROHIBITED,
                RedistributionPolicy.UNKNOWN,
            }:
                reasons.add(f"{document.document_id}:{document.redistribution.value}")
            if document.access_class in {
                AccessClass.CREDENTIALLED,
                AccessClass.DUA,
                AccessClass.LOCAL_PRIVATE,
                AccessClass.QUARANTINE,
            }:
                reasons.add(f"{document.document_id}:{document.access_class.value}")
        return PolicyDecision(not reasons, tuple(sorted(reasons)))

    def artifact_storage(
        self,
        source: SourceDefinition,
        *,
        store_uri: str,
        encrypted_at_rest: bool,
    ) -> PolicyDecision:
        """Validate storage placement before restricted bytes are fetched."""

        reasons: list[str] = []
        remote = "://" in store_uri and not store_uri.startswith("file://")
        if source.retention is RetentionPolicy.LOCAL_ONLY and remote:
            reasons.append("local_only_source_requires_local_store")
        if source.access_class in {AccessClass.DUA, AccessClass.LOCAL_PRIVATE}:
            if not encrypted_at_rest:
                reasons.append("restricted_source_requires_encryption_at_rest")
        return PolicyDecision(not reasons, tuple(reasons))


class MiningQualityGate:
    """Validate mined data independently from task-specific prediction schemas."""

    def __init__(
        self,
        *,
        known_concepts: Collection[tuple[str, str, str]] | None = None,
    ) -> None:
        self.known_concepts = set(known_concepts) if known_concepts is not None else None

    def validate(
        self,
        documents: Sequence[MinedDocument],
        annotations: Sequence[AnnotationProposal],
    ) -> tuple[str, ...]:
        issues: list[str] = []
        documents_by_id = {document.document_id: document for document in documents}
        duplicate_documents = _duplicates(document.document_id for document in documents)
        duplicate_annotations = _duplicates(annotation.annotation_id for annotation in annotations)
        issues.extend(f"duplicate_document:{value}" for value in duplicate_documents)
        issues.extend(f"duplicate_annotation:{value}" for value in duplicate_annotations)
        for annotation in annotations:
            document = documents_by_id.get(annotation.document_id)
            if document is None:
                issues.append(f"missing_document:{annotation.annotation_id}")
                continue
            try:
                annotation.validate_offsets(document)
            except ValueError as error:
                issues.append(f"offset:{annotation.annotation_id}:{error}")
            if annotation.layer in {AnnotationLayer.GOLD, AnnotationLayer.CHALLENGE}:
                if annotation.review_status is not ReviewStatus.ACCEPTED:
                    issues.append(f"unreviewed_{annotation.layer.value}:{annotation.annotation_id}")
            if annotation.layer is AnnotationLayer.CHALLENGE:
                if annotation.metadata.get("origin") == "synthetic":
                    issues.append(f"synthetic_challenge:{annotation.annotation_id}")
            if self.known_concepts is not None:
                for concept in annotation.concepts:
                    key = (concept.code_system, concept.code, concept.terminology_version)
                    if key not in self.known_concepts:
                        issues.append(
                            "unknown_concept:"
                            f"{annotation.annotation_id}:{concept.code_system}:{concept.code}"
                        )
        return tuple(sorted(issues))


def _duplicates(values: Iterable[str]) -> tuple[str, ...]:
    counts: Counter[str] = Counter(values)
    return tuple(sorted(value for value, count in counts.items() if count > 1))
