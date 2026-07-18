"""Concept-linked source annotation alias-mining tests."""

from __future__ import annotations

from dataclasses import replace

from medical_kg_nlp.mining.linked_aliases import (
    LinkedAliasProposalPolicy,
    build_linked_alias_proposals,
)
from medical_kg_nlp.mining.records import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    ConceptLink,
    MinedDocument,
    RedistributionPolicy,
    ReviewStatus,
    SourceArtifact,
    StoredObject,
)
from medical_kg_nlp.schema.types import CodeSystem, EntityType

_SOURCE_SHA256 = "b" * 64


def _artifact() -> SourceArtifact:
    return SourceArtifact(
        artifact_id="codiesp:fixture",
        source_id="codiesp",
        source_version="fixture-v1",
        source_uri="https://example.test/codiesp.zip",
        object=StoredObject(
            sha256=_SOURCE_SHA256,
            uri="objects/fixture",
            byte_size=100,
        ),
        media_type="application/zip",
        license_id="CC-BY-4.0",
        access_class=AccessClass.OPEN,
        redistribution=RedistributionPolicy.ATTRIBUTION,
        hosted_processing_allowed=True,
        retrieved_at="2026-07-18T00:00:00Z",
    )


def _document(document_id: str, *, corpus_split: str | None = None) -> MinedDocument:
    return MinedDocument(
        document_id=document_id,
        text="sinovitis",
        language="es",
        note_type="clinical_case",
        source_artifact_id="codiesp:fixture",
        access_class=AccessClass.OPEN,
        redistribution=RedistributionPolicy.ATTRIBUTION,
        hosted_processing_allowed=True,
        metadata={"corpus_split": corpus_split} if corpus_split is not None else {},
    )


def _annotation(document_id: str, annotation_id: str, *, code: str = "M65.9") -> AnnotationProposal:
    return AnnotationProposal(
        annotation_id=annotation_id,
        document_id=document_id,
        span=(0, 9),
        text="sinovitis",
        entity_type="FINDING",
        assertions=(),
        concepts=(
            ConceptLink(
                code_system="ICD-10-CM",
                code=code,
                terminology_version="codiesp-v1.4",
            ),
        ),
        confidence=1.0,
        layer=AnnotationLayer.SILVER,
        label_source="source_human_annotation",
        labeler_id="codiesp-fixture",
        review_status=ReviewStatus.PROPOSED,
        source_label="DIAGNOSTICO",
        metadata={"discontinuous": "false", "source_text_match": "true"},
    )


def _policy() -> LinkedAliasProposalPolicy:
    return LinkedAliasProposalPolicy(
        policy_id="codiesp-linked-alias-v1",
        accepted_source_ids=("codiesp",),
        accepted_source_sha256=(_SOURCE_SHA256,),
        source_code_systems=(("ICD-10-CM", CodeSystem.ICD10),),
        source_label_types=(("DIAGNOSTICO", EntityType.DISEASE),),
        accepted_label_sources=("source_human_annotation",),
        accepted_layers=("silver",),
        accepted_review_statuses=("proposed",),
        proposal_review_status="source_human_annotation",
        alias_tty="SOURCE_HUMAN_ANNOTATION",
    )


def test_linked_alias_builder_aggregates_multi_document_human_links() -> None:
    documents = (_document("doc-1"), _document("doc-2"))
    annotations = (
        _annotation("doc-1", "annotation-1"),
        _annotation("doc-2", "annotation-2"),
    )

    result = build_linked_alias_proposals(documents, annotations, (_artifact(),), _policy())

    assert len(result.proposals) == 1
    proposal = result.proposals[0]
    assert proposal["normalized_alias"] == "sinovitis"
    assert proposal["code_system"] == "ICD-10"
    assert proposal["code"] == "M65.9"
    assert proposal["semantic_type"] == "DISEASE"
    assert proposal["supporting_record_count"] == 2
    assert result.report["proposal_contract"].startswith("source-pinned")


def test_linked_alias_builder_rejects_discontinuous_and_conflicting_links() -> None:
    documents = (_document("doc-1"), _document("doc-2"), _document("doc-3"))
    discontinuous = replace(
        _annotation("doc-1", "annotation-discontinuous"),
        metadata={"discontinuous": "true", "source_text_match": "true"},
    )
    conflicting = (
        _annotation("doc-2", "annotation-code-1", code="M65.9"),
        _annotation("doc-3", "annotation-code-2", code="M65.8"),
    )

    result = build_linked_alias_proposals(
        documents,
        (discontinuous, *conflicting),
        (_artifact(),),
        _policy(),
    )

    assert not result.proposals
    assert result.report["reason_counts"] == {
        "discontinuous_span": 1,
        "source_target_conflict": 2,
    }


def test_linked_alias_policy_filters_official_source_split() -> None:
    documents = (
        _document("train-1", corpus_split="train"),
        _document("train-2", corpus_split="train"),
        _document("dev-1", corpus_split="dev"),
    )
    annotations = (
        _annotation("train-1", "annotation-train-1"),
        _annotation("train-2", "annotation-train-2"),
        _annotation("dev-1", "annotation-dev-1"),
    )
    policy = replace(
        _policy(),
        document_metadata_filters=(("corpus_split", ("train",)),),
    )

    result = build_linked_alias_proposals(documents, annotations, (_artifact(),), policy)

    assert result.proposals[0]["supporting_record_count"] == 2
    assert result.report["document_metadata_filters"] == {"corpus_split": ["train"]}
    assert result.report["reason_counts"]["document_metadata_not_allowed"] == 1
