"""Stage orchestration for an already composed clinical NLP pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from medical_kg_nlp.governance.audit import AuditEvent

from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.pipeline.components import PipelineComponents
from medical_kg_nlp.pipeline.ports import (
    BatchCandidateRerankerPort,
    BatchCandidateRetrieverPort,
    CandidateRerankRequest,
    CandidateRetrievalRequest,
)
from medical_kg_nlp.pipeline.runtime import Closable, RuntimeCapabilities
from medical_kg_nlp.pipeline.stages import (
    AssertionClassificationStage,
    DocumentPreparationStage,
    EntityExtractionStage,
    PredictionValidationStage,
)
from medical_kg_nlp.pipeline.tracing import PipelineTrace
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.document import ClinicalDocument
from medical_kg_nlp.schema.output import ClinicalPrediction
from medical_kg_nlp.schema.types import CodeSystem
from medical_kg_nlp.schema.validator import PredictionValidator
from medical_kg_nlp.utils.hashing import sha256_text
from medical_kg_nlp.utils.text import text_window

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

        with trace.stage("entity_extraction") as counters:
            entities = self._entity_stage.run(loaded_document.text)
            counters["entities"] = len(entities)

        with trace.stage("context_assertion_classification") as counters:
            if self.options.enable_context:
                self._assertion_stage.run(entities, tuple(sentences), counters)
            else:
                counters["skipped_entities"] = len(entities)

        contexts_by_entity: dict[str, str] = {}
        mentions_by_entity: dict[str, str] = {}
        generated_candidates: dict[str, list[Candidate]] = {}
        with trace.stage("candidate_generation") as counters:
            if self.options.enable_linking:
                retriever = self.components.candidate_retriever
                if retriever is None:
                    raise RuntimeError("Candidate retriever component is unavailable.")
                generated_total = 0
                entities_with_candidates = 0
                pinned_entities = 0
                retrieval_requests: list[CandidateRetrievalRequest] = []
                for entity in entities:
                    if entity.code_system != CodeSystem.NONE:
                        pinned_entities += 1
                    context = text_window(
                        loaded_document.text, entity.span, radius=self.options.context_window
                    )
                    contexts_by_entity[entity.id] = context
                    mention = _linking_mention(loaded_document.text, entity)
                    mentions_by_entity[entity.id] = mention
                    retrieval_requests.append(
                        CandidateRetrievalRequest(
                            entity_id=entity.id,
                            entity_type=entity.type,
                            mention=mention,
                            context_window=context,
                        )
                    )
                if isinstance(retriever, BatchCandidateRetrieverPort):
                    retrieved_by_entity = retriever.retrieve_batch(tuple(retrieval_requests))
                    _validate_batch_keys(
                        tuple(request.entity_id for request in retrieval_requests),
                        retrieved_by_entity,
                        stage="candidate retrieval",
                    )
                else:
                    retrieved_by_entity = {
                        request.entity_id: retriever.retrieve(
                            entities[index],
                            request.context_window,
                            request.mention,
                        )
                        for index, request in enumerate(retrieval_requests)
                    }
                for request in retrieval_requests:
                    candidates = list(retrieved_by_entity.get(request.entity_id, []))
                    generated_candidates[request.entity_id] = candidates
                    generated_total += len(candidates)
                    entities_with_candidates += int(bool(candidates))
                    bucket = min(len(candidates), 10)
                    key = f"candidate_count_{bucket}"
                    counters[key] = counters.get(key, 0) + 1
                    for candidate in candidates:
                        for source in candidate.sources:
                            key = f"source_{source}"
                            counters[key] = counters.get(key, 0) + 1
                counters["candidate_entities"] = len(generated_candidates)
                counters["pinned_entities"] = pinned_entities
                counters["entities_with_candidates"] = entities_with_candidates
                counters["generated_candidates"] = generated_total
                counters["candidate_sources"] = len(self.options.candidate_sources)
            else:
                counters["skipped_entities"] = len(entities)

        reranked_candidates: dict[str, list[Candidate]] = {}
        entities_by_id = {entity.id: entity for entity in entities}
        with trace.stage("candidate_reranking") as counters:
            if self.options.enable_linking:
                reranker = self.components.candidate_reranker
                if self.options.enable_candidate_reranking and reranker is None:
                    raise RuntimeError("Candidate reranker component is unavailable.")
                rerank_requests = tuple(
                    CandidateRerankRequest(
                        entity_id=entity_id,
                        mention=mentions_by_entity.get(
                            entity_id,
                            _linking_mention(loaded_document.text, entities_by_id[entity_id]),
                        ),
                        context_window=contexts_by_entity.get(entity_id, ""),
                        candidates=tuple(candidates),
                    )
                    for entity_id, candidates in generated_candidates.items()
                )
                if self.options.enable_candidate_reranking and isinstance(
                    reranker, BatchCandidateRerankerPort
                ):
                    reranked_by_entity = reranker.rerank_batch(rerank_requests)
                    _validate_batch_keys(
                        tuple(request.entity_id for request in rerank_requests),
                        reranked_by_entity,
                        stage="candidate reranking",
                    )
                else:
                    reranked_by_entity = {
                        request.entity_id: (
                            reranker.rerank(
                                list(request.candidates),
                                request.context_window,
                                request.mention,
                            )
                            if self.options.enable_candidate_reranking and reranker is not None
                            else list(request.candidates)
                        )
                        for request in rerank_requests
                    }
                for rerank_request in rerank_requests:
                    ranked = reranked_by_entity.get(rerank_request.entity_id)
                    reranked_candidates[rerank_request.entity_id] = list(
                        rerank_request.candidates if ranked is None else ranked
                    )
                counters["reranked_entities"] = len(reranked_candidates)
                counters["reranked_candidates"] = sum(
                    len(candidates) for candidates in reranked_candidates.values()
                )
                counters["skipped_reranking"] = (
                    0 if self.options.enable_candidate_reranking else len(reranked_candidates)
                )
                stats = getattr(reranker, "stats", None)
                if callable(stats):
                    for name, value in stats().items():
                        if isinstance(value, int):
                            counters[name] = value
            else:
                counters["skipped_entities"] = len(entities)

        with trace.stage("graph_evidence_reranking") as counters:
            if self.options.enable_linking and self.options.enable_graph_evidence_reranking:
                document_reranker = self.components.document_candidate_reranker
                if document_reranker is None:
                    raise RuntimeError("Document candidate reranker component is unavailable.")
                reranked_candidates, graph_counters = document_reranker.rerank_document(
                    entities,
                    reranked_candidates,
                    sentences,
                    mentions_by_entity,
                )
                counters.update(graph_counters)
            elif self.options.enable_linking:
                counters["skipped_entities"] = len(reranked_candidates)
            else:
                counters["skipped_entities"] = len(entities)

        with trace.stage("normalization_assignment") as counters:
            if self.options.enable_linking:
                assigner = self.components.candidate_assigner
                if assigner is None:
                    raise RuntimeError("Candidate assigner component is unavailable.")
                for entity in entities:
                    assigned_candidates = reranked_candidates.get(entity.id)
                    if assigned_candidates is None:
                        continue
                    assigner.assign(
                        entity,
                        assigned_candidates,
                        mention=_linking_mention(loaded_document.text, entity),
                    )
                counters["assigned_codes"] = sum(entity.code is not None for entity in entities)
                counters["unlinked_entities"] = sum(entity.code is None for entity in entities)
                counters["qualified_candidates"] = sum(
                    candidate.qualified
                    for entity in entities
                    for candidate in entity.candidates
                )
                counters["entities_with_qualified_candidates"] = sum(
                    any(candidate.qualified for candidate in entity.candidates)
                    for entity in entities
                )
                for entity in entities:
                    for schema_candidate in entity.candidates:
                        reason = schema_candidate.qualification_reason or "unspecified"
                        key = f"qualification_{reason}"
                        counters[key] = counters.get(key, 0) + 1
            else:
                counters["skipped_entities"] = len(entities)

        with trace.stage("icd_rxnorm_umls_validation") as counters:
            if self.options.enable_entity_kg_validation:
                validator = self.components.knowledge_validator
                if validator is None:
                    raise RuntimeError("Knowledge validator component is unavailable.")
                entities, entity_issues = validator.validate_entities(entities)
                counters["issues"] = len(entity_issues)
            else:
                counters["skipped_entities"] = len(entities)

        with trace.stage("relation_extraction") as counters:
            if self.options.enable_relations:
                extractor = self.components.relation_extractor
                if extractor is None:
                    raise RuntimeError("Relation extractor component is unavailable.")
                relations = extractor.extract(entities, sentences)
                counters["relations"] = len(relations)
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

        with trace.stage("structured_json_output") as counters:
            prediction = ClinicalPrediction.from_text(
                document_id=loaded_document.document_id,
                text=loaded_document.text,
                entities=entities,
                relations=relations,
                pipeline_version=self.components.pipeline_version,
            )
            counters["entities"] = len(prediction.entities)
            counters["relations"] = len(prediction.relations)

        with trace.stage("prediction_validation") as counters:
            validation = self._validation_stage.run(
                prediction,
                source_text=loaded_document.text,
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
                # INVARIANT: component adapters cannot bypass offset, type/code-system,
                # duplicate-ID, or relation safety by disabling optional KG stages.
                detail = "; ".join(
                    f"{item.issue.kind} at {item.issue.path}: {item.issue.message}"
                    for item in errors
                )
                raise ValueError(f"Core prediction validation failed: {detail}")
        trace.mark_finished(
            success=True,
            entities=len(prediction.entities),
            assigned_codes=sum(entity.code is not None for entity in prediction.entities),
        )
        return PipelineRunResult(prediction=prediction, trace=trace)

def _linking_mention(source_text: str, entity: EntityAnnotation) -> str:
    medication = entity.medication_mention
    if medication is None:
        return entity.text
    start, end = medication.full_span
    return source_text[start:end]


def _validate_batch_keys(
    expected_ids: tuple[str, ...],
    results: dict[str, list[Candidate]],
    *,
    stage: str,
) -> None:
    """Reject incomplete batch responses before they can alter entity alignment."""

    expected = set(expected_ids)
    actual = set(results)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{stage} batch returned inconsistent entity IDs: "
            f"missing={missing}, unexpected={unexpected}"
        )
