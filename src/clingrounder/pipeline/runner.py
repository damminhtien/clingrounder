"""Stage orchestration for an already composed clinical NLP pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from clingrounder.governance.audit import AuditEvent

from clingrounder.pipeline.components import PipelineComponents
from clingrounder.pipeline.runtime import Closable, RuntimeCapabilities
from clingrounder.pipeline.stages import (
    AssertionClassificationStage,
    CandidateGenerationStage,
    CandidateRerankingStage,
    DocumentPreparationStage,
    EntityKnowledgeValidationStage,
    EntityExtractionStage,
    GraphEvidenceRerankingStage,
    LinkingContext,
    LinkingStageResult,
    DocumentStructure,
    NormalizationAssignmentStage,
    PreparedDocument,
    PredictionValidationStage,
    RelationExtractionStage,
)
from clingrounder.pipeline.tracing import PipelineTrace
from clingrounder.schema.annotation import EntityAnnotation, RelationAnnotation
from clingrounder.schema.document import ClinicalDocument
from clingrounder.schema.output import ClinicalPrediction
from clingrounder.schema.validator import PredictionValidator
from clingrounder.utils.hashing import sha256_text

__all__ = ["PipelineRunResult", "PipelineRunner"]


@dataclass(frozen=True)
class PipelineRunResult:
    """Prediction and machine-readable stage trace for one document."""

    prediction: ClinicalPrediction
    trace: PipelineTrace


class PipelineRunner:
    """Orchestrate stages using only injected components."""

    def __init__(self, components: PipelineComponents) -> None:
        self.components = components
        self.options = components.options
        self._resources: tuple[Closable, ...] = ()
        self._closed = False
        self._document_stage = DocumentPreparationStage()
        self._entity_stage = EntityExtractionStage(components.entity_extractor)
        self._assertion_stage = AssertionClassificationStage(components.assertion_classifier)
        self._candidate_generation_stage = CandidateGenerationStage(
            components.candidate_retriever,
            self.options.context_window,
        )
        self._candidate_reranking_stage = CandidateRerankingStage(
            components.candidate_reranker,
            self.options.enable_candidate_reranking,
        )
        self._graph_evidence_stage = GraphEvidenceRerankingStage(
            components.document_candidate_reranker
        )
        self._assignment_stage = NormalizationAssignmentStage(components.candidate_assigner)
        self._entity_validation_stage = EntityKnowledgeValidationStage(
            components.knowledge_validator
        )
        self._relation_stage = RelationExtractionStage(components.relation_extractor)
        self._validation_stage = PredictionValidationStage(
            PredictionValidator(components.terminology_repository)
        )

    def attach_resources(self, resources: tuple[Closable, ...]) -> None:
        """Attach composition-owned resources without changing the runner contract."""

        if self._closed:
            raise RuntimeError("Cannot attach resources to a closed PipelineRunner")
        if self._resources:
            raise RuntimeError("PipelineRunner resources are already attached")
        self._resources = resources

    @property
    def resources(self) -> tuple[Closable, ...]:
        """Expose immutable composition resources to :class:`PipelineRuntime`."""

        return self._resources

    def close(self) -> None:
        """Close composed resources in reverse order; repeated calls are harmless."""

        if self._closed:
            return
        self._closed = True
        seen: set[int] = set()
        for resource in reversed(self._resources):
            if id(resource) in seen:
                continue
            seen.add(id(resource))
            resource.close()

    @property
    def runtime_capabilities(self) -> RuntimeCapabilities:
        """Expose the composed component policy to batch orchestration."""

        return self.components.runtime_capabilities

    def process_text(
        self,
        document_id: str,
        text: str,
        metadata: dict[str, str] | None = None,
    ) -> ClinicalPrediction:
        return self.process_text_with_trace(document_id, text, metadata).prediction

    def process_text_with_trace(
        self,
        document_id: str,
        text: str,
        metadata: dict[str, str] | None = None,
    ) -> PipelineRunResult:
        document = ClinicalDocument(document_id=document_id, text=text, metadata=metadata or {})
        return self.process_document_with_trace(document)

    def process_document(self, document: ClinicalDocument) -> ClinicalPrediction:
        return self.process_document_with_trace(document).prediction

    def process_document_with_trace(self, document: ClinicalDocument) -> PipelineRunResult:
        self.components.audit_sink.emit(
            AuditEvent(
                "prediction",
                document_id_hash=sha256_text(document.document_id),
                profile_fingerprint=self.components.configuration_fingerprint,
                model_revision=self.components.model_revision,
                terminology_fingerprint=self.components.terminology_fingerprint,
                details={"document_length": str(len(document.text))},
            )
        )
        trace = PipelineTrace(
            document_id=document.document_id,
            observer=self.components.observer,
            document_length=len(document.text),
            configuration_fingerprint=self.components.configuration_fingerprint,
            terminology_fingerprint=self.components.terminology_fingerprint,
            model_revision=self.components.model_revision,
            backend=self.components.backend,
            worker=self.components.worker,
            redact_errors=not self.components.trace_include_error_messages,
        )
        try:
            return self._process_document_with_trace(document, trace)
        except BaseException as error:
            self.components.audit_sink.emit(
                AuditEvent(
                    "validation_failure",
                    outcome="failure",
                    document_id_hash=sha256_text(document.document_id),
                    profile_fingerprint=self.components.configuration_fingerprint,
                    details={"error_type": type(error).__name__},
                )
            )
            trace.mark_finished(success=False)
            trace.attach_to(error)
            raise

    def _process_document_with_trace(
        self,
        document: ClinicalDocument,
        trace: PipelineTrace,
    ) -> PipelineRunResult:
        prepared = self._prepare_document(document, trace)
        entities = self._extract_entities(prepared, trace)
        self._link_entities(prepared, entities, trace)
        entities = self._validate_entities(entities, trace)
        relations = self._extract_relations(entities, prepared, trace)
        prediction = self._build_prediction(prepared, entities, relations, trace)
        self._validate_prediction(prediction, prepared, trace)
        trace.mark_finished(
            success=True,
            entities=len(prediction.entities),
            assigned_codes=sum(entity.code is not None for entity in prediction.entities),
        )
        return PipelineRunResult(prediction=prediction, trace=trace)

    def _prepare_document(
        self,
        document: ClinicalDocument,
        trace: PipelineTrace,
    ) -> PreparedDocument:
        """Load and segment a document while preserving the existing trace contract."""

        with trace.stage("document_loader") as counters:
            loaded_document = document
            counters["documents"] = 1
            counters["characters"] = len(loaded_document.text)
            counters["metadata_fields"] = len(loaded_document.metadata)

        with trace.stage("lookup_normalization_diagnostics") as counters:
            if self.options.enable_lookup_normalization_diagnostics:
                counters.update(
                    self._document_stage.diagnostics(
                        loaded_document.text,
                        self.components.normalization_contract,
                    )
                )
            else:
                counters["skipped"] = 1

        with trace.stage("section_detection") as counters:
            structure = self._document_stage.structure(loaded_document.text)
            sections = list(structure.sections)
            counters["sections"] = len(sections)

        with trace.stage("sentence_splitting") as counters:
            sentences = list(structure.sentences)
            counters["sentences"] = len(sentences)
        return PreparedDocument(
            document=loaded_document,
            structure=DocumentStructure(tuple(sections), tuple(sentences)),
        )

    def _extract_entities(
        self,
        prepared: PreparedDocument,
        trace: PipelineTrace,
    ) -> list[EntityAnnotation]:
        """Extract raw-coordinate entities and apply optional context assertions."""

        with trace.stage("entity_extraction") as counters:
            entities = self._entity_stage.run(prepared.document.text)
            counters["entities"] = len(entities)

        with trace.stage("context_assertion_classification") as counters:
            if self.options.enable_context:
                self._assertion_stage.run(entities, prepared.sentences, counters)
            else:
                counters["skipped_entities"] = len(entities)
        return entities

    def _link_entities(
        self,
        prepared: PreparedDocument,
        entities: list[EntityAnnotation],
        trace: PipelineTrace,
    ) -> LinkingStageResult:
        """Retrieve, rerank, and assign candidates without changing stage order."""

        linking = LinkingStageResult(
            context=LinkingContext(mentions_by_entity={}, contexts_by_entity={}),
            generated_candidates={},
            reranked_candidates={},
        )
        with trace.stage("candidate_generation") as counters:
            if self.options.enable_linking:
                generated = self._candidate_generation_stage.run(
                    prepared.document.text, entities
                )
                linking = LinkingStageResult(
                    context=LinkingContext(
                        mentions_by_entity=generated.mentions_by_entity,
                        contexts_by_entity=generated.contexts_by_entity,
                    ),
                    generated_candidates=generated.candidates_by_entity,
                    reranked_candidates={},
                )
                counters.update(generated.counters)
                counters["candidate_sources"] = len(self.options.candidate_sources)
            else:
                counters["skipped_entities"] = len(entities)

        with trace.stage("candidate_reranking") as counters:
            if self.options.enable_linking:
                reranked = self._candidate_reranking_stage.run(
                    entities,
                    linking.generated_candidates,
                    linking.context.contexts_by_entity,
                    linking.context.mentions_by_entity,
                )
                linking = LinkingStageResult(
                    context=linking.context,
                    generated_candidates=linking.generated_candidates,
                    reranked_candidates=reranked.candidates_by_entity,
                )
                counters.update(reranked.counters)
            else:
                counters["skipped_entities"] = len(entities)

        with trace.stage("graph_evidence_reranking") as counters:
            if self.options.enable_linking and self.options.enable_graph_evidence_reranking:
                reranked_candidates, graph_counters = self._graph_evidence_stage.run(
                    entities,
                    linking.reranked_candidates,
                    list(prepared.sentences),
                    linking.context.mentions_by_entity,
                )
                linking = LinkingStageResult(
                    context=linking.context,
                    generated_candidates=linking.generated_candidates,
                    reranked_candidates=reranked_candidates,
                )
                counters.update(graph_counters)
            elif self.options.enable_linking:
                counters["skipped_entities"] = len(linking.reranked_candidates)
            else:
                counters["skipped_entities"] = len(entities)

        with trace.stage("normalization_assignment") as counters:
            if self.options.enable_linking:
                counters.update(
                    self._assignment_stage.run(
                        prepared.document.text,
                        entities,
                        linking.reranked_candidates,
                    )
                )
            else:
                counters["skipped_entities"] = len(entities)
        return linking

    def _validate_entities(
        self,
        entities: list[EntityAnnotation],
        trace: PipelineTrace,
    ) -> list[EntityAnnotation]:
        """Run entity/terminology validation and preserve its trace counters."""

        with trace.stage("icd_rxnorm_umls_validation") as counters:
            if self.options.enable_entity_kg_validation:
                entity_validation = self._entity_validation_stage.run(entities)
                counters["issues"] = len(entity_validation.issues)
                return entity_validation.entities
            counters["skipped_entities"] = len(entities)
        return entities

    def _extract_relations(
        self,
        entities: list[EntityAnnotation],
        prepared: PreparedDocument,
        trace: PipelineTrace,
    ) -> list[RelationAnnotation]:
        """Extract and validate relations after entity validation."""

        with trace.stage("relation_extraction") as counters:
            if self.options.enable_relations:
                extracted = self._relation_stage.run(entities, list(prepared.sentences))
                relations = extracted.relations
                counters.update(extracted.counters)
            else:
                relations = []
                counters["skipped_entities"] = len(entities)

        with trace.stage("ontology_kg_consistency_check") as counters:
            if self.options.enable_relation_kg_validation:
                validator = self.components.knowledge_validator
                if validator is None:
                    raise RuntimeError("Knowledge validator component is unavailable.")
                relations, relation_issues = validator.validate_relations(entities, relations)
                counters["issues"] = len(relation_issues)
                counters["relations"] = len(relations)
            else:
                counters["relations"] = len(relations)
        return relations

    def _build_prediction(
        self,
        prepared: PreparedDocument,
        entities: list[EntityAnnotation],
        relations: list[RelationAnnotation],
        trace: PipelineTrace,
    ) -> ClinicalPrediction:
        """Construct the published prediction while retaining output counters."""

        with trace.stage("structured_json_output") as counters:
            prediction = ClinicalPrediction.from_text(
                document_id=prepared.document.document_id,
                text=prepared.document.text,
                entities=entities,
                relations=relations,
                pipeline_version=self.components.pipeline_version,
            )
            counters["entities"] = len(prediction.entities)
            counters["relations"] = len(prediction.relations)
        return prediction

    def _validate_prediction(
        self,
        prediction: ClinicalPrediction,
        prepared: PreparedDocument,
        trace: PipelineTrace,
    ) -> None:
        """Apply final core validation and raise without discarding the partial trace."""

        with trace.stage("prediction_validation") as counters:
            validation = self._validation_stage.run(
                prediction,
                source_text=prepared.document.text,
            )
            profiled_issues = list(validation.issues)
            errors = list(validation.errors)
            counters["issues"] = len(profiled_issues)
            counters["errors"] = len(errors)
            counters["warnings"] = len(profiled_issues) - len(errors)
            counters["validated_entities"] = len(prediction.entities)
            counters["validated_relations"] = len(prediction.relations)
            for item in profiled_issues:
                key = f"validation_{item.issue.kind}"
                counters[key] = counters.get(key, 0) + 1
            if errors:
                # INVARIANT: adapters cannot bypass offset, type/code-system, duplicate-ID,
                # or relation safety by disabling optional KG stages.
                detail = "; ".join(
                    f"{item.issue.kind} at {item.issue.path}: {item.issue.message}"
                    for item in errors
                )
                raise ValueError(f"Core prediction validation failed: {detail}")
