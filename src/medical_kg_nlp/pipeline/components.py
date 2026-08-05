"""Dependency container consumed by the pipeline runner."""

from __future__ import annotations

from dataclasses import dataclass

from medical_kg_nlp.pipeline.options import PipelineOptions
from medical_kg_nlp.pipeline.runtime import RuntimeCapabilities
from medical_kg_nlp.pipeline.ports import (
    AssertionClassifierPort,
    CandidateAssignerPort,
    CandidateRerankerPort,
    DocumentCandidateRerankerPort,
    CandidateRetrieverPort,
    EntityExtractorPort,
    KnowledgeValidatorPort,
    RelationExtractorPort,
    TerminologyRepository,
)
from medical_kg_nlp.preprocessing.normalizer import (
    DEFAULT_NORMALIZATION_CONTRACT,
    NormalizationContract,
)

__all__ = ["PipelineComponents"]


@dataclass(frozen=True)
class PipelineComponents:
    """Concrete components and immutable runtime settings for one runner."""

    entity_extractor: EntityExtractorPort
    assertion_classifier: AssertionClassifierPort | None = None
    candidate_retriever: CandidateRetrieverPort | None = None
    candidate_reranker: CandidateRerankerPort | None = None
    document_candidate_reranker: DocumentCandidateRerankerPort | None = None
    candidate_assigner: CandidateAssignerPort | None = None
    relation_extractor: RelationExtractorPort | None = None
    knowledge_validator: KnowledgeValidatorPort | None = None
    terminology_repository: TerminologyRepository | None = None
    options: PipelineOptions = PipelineOptions()
    runtime_capabilities: RuntimeCapabilities = RuntimeCapabilities()
    normalization_contract: NormalizationContract = DEFAULT_NORMALIZATION_CONTRACT
    pipeline_version: str = "0.2.0"

    def __post_init__(self) -> None:
        if self.options.enable_context and self.assertion_classifier is None:
            raise ValueError("enable_context requires an assertion_classifier component")
        if self.options.enable_linking:
            if self.candidate_retriever is None:
                raise ValueError("enable_linking requires a candidate_retriever component")
            if self.candidate_assigner is None:
                raise ValueError("enable_linking requires a candidate_assigner component")
            if self.options.enable_candidate_reranking and self.candidate_reranker is None:
                raise ValueError(
                    "enable_candidate_reranking requires a candidate_reranker component"
                )
            if (
                self.options.enable_graph_evidence_reranking
                and self.document_candidate_reranker is None
            ):
                raise ValueError(
                    "enable_graph_evidence_reranking requires a "
                    "document_candidate_reranker component"
                )
        if self.options.enable_relations and self.relation_extractor is None:
            raise ValueError("enable_relations requires a relation_extractor component")
        if (
            self.options.enable_entity_kg_validation
            or self.options.enable_relation_kg_validation
        ) and self.knowledge_validator is None:
            raise ValueError("KG validation requires a knowledge_validator component")
