from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import pytest

from medical_kg_nlp.datasets.synthetic_adapter import SyntheticDatasetAdapter
from medical_kg_nlp.pipeline import (
    InMemoryPipelineObserver,
    NoOpPipelineObserver,
    PipelineComponents,
    PipelineFactory,
    PipelineFactoryConfig,
    PipelineOptions,
    PipelineRunner,
)
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.types import AssertionStatus


def test_pipeline_trace_records_algorithm_stages() -> None:
    document = SyntheticDatasetAdapter().load_documents("data/samples/sample_notes.jsonl")[0]
    result = PipelineFactory.from_config().process_document_with_trace(document)

    stage_names = [stage.name for stage in result.trace.stages]
    assert stage_names == [
        "document_loader",
        "lookup_normalization_diagnostics",
        "section_detection",
        "sentence_splitting",
        "entity_extraction",
        "context_assertion_classification",
        "candidate_generation",
        "candidate_reranking",
        "graph_evidence_reranking",
        "normalization_assignment",
        "icd_rxnorm_umls_validation",
        "relation_extraction",
        "ontology_kg_consistency_check",
        "structured_json_output",
        "prediction_validation",
    ]
    assert result.trace.total_ms >= 0
    assert result.trace.bottleneck() is not None
    assert result.trace.to_json()["bottleneck_stage"] in stage_names
    assert result.prediction.document_id == document.document_id
    by_stage = {stage.name: stage for stage in result.trace.stages}
    normalization = by_stage["lookup_normalization_diagnostics"].counters
    assert normalization["offset_map_entries"] > 0
    assert normalization["source_coordinate_spans"] == 1
    assert normalization["normalized_text_used_downstream"] == 0
    candidate_counters = by_stage["candidate_generation"].counters
    assert candidate_counters["generated_candidates"] + candidate_counters["pinned_entities"] > 0
    assert by_stage["graph_evidence_reranking"].counters["skipped_entities"] > 0
    assignment_counters = by_stage["normalization_assignment"].counters
    assert assignment_counters["qualified_candidates"] >= 0
    assert assignment_counters["entities_with_qualified_candidates"] >= 0
    assert by_stage["normalization_assignment"].counters["assigned_codes"] > 0


def test_pipeline_options_can_disable_context_and_relations() -> None:
    document = SyntheticDatasetAdapter().load_documents("data/samples/sample_notes.jsonl")[0]
    runner = PipelineFactory.from_config(
        PipelineFactoryConfig(
            options=PipelineOptions(enable_context=False, enable_relations=False)
        )
    )
    result = runner.process_document_with_trace(document)

    by_text = {entity.text: entity for entity in result.prediction.entities}
    assert by_text["viêm phổi"].assertion == AssertionStatus.UNKNOWN
    assert result.prediction.relations == []
    relation_stage = next(
        stage for stage in result.trace.stages if stage.name == "relation_extraction"
    )
    assert relation_stage.counters["skipped_entities"] == len(result.prediction.entities)


def test_pipeline_can_process_raw_text_and_skip_reranking() -> None:
    document = SyntheticDatasetAdapter().load_documents("data/samples/sample_notes.jsonl")[0]
    runner = PipelineFactory.from_config(
        PipelineFactoryConfig(
            options=PipelineOptions(enable_candidate_reranking=False)
        )
    )
    result = runner.process_text_with_trace(document.document_id, document.text, document.metadata)

    by_stage = {stage.name: stage for stage in result.trace.stages}
    assert by_stage["document_loader"].counters["documents"] == 1
    skipped = by_stage["candidate_reranking"].counters["skipped_reranking"]
    pinned = by_stage["candidate_generation"].counters["pinned_entities"]
    assert skipped + pinned > 0
    assert result.prediction.document_id == document.document_id


def test_entity_only_runner_does_not_build_linking_indexes() -> None:
    runner = PipelineFactory.from_config(
        PipelineFactoryConfig(
            options=PipelineOptions(
                enable_context=False,
                enable_linking=False,
                enable_candidate_reranking=False,
                enable_entity_kg_validation=False,
                enable_relations=False,
                enable_relation_kg_validation=False,
            )
        )
    )

    assert runner.components.candidate_retriever is None
    result = runner.process_text_with_trace("entity-only", "Bệnh nhân ho.")
    by_stage = {stage.name: stage for stage in result.trace.stages}
    assert by_stage["candidate_generation"].counters["skipped_entities"] > 0


def test_trace_is_phi_safe_and_contains_runtime_metadata() -> None:
    observer = InMemoryPipelineObserver()
    base = PipelineFactory.from_config()
    runner = PipelineRunner(replace(base.components, observer=observer))
    result = runner.process_text_with_trace("patient-123", "Bệnh nhân ho.")

    payload = result.trace.to_json()
    assert "patient-123" not in str(payload)
    assert "Bệnh nhân" not in str(payload)
    assert result.trace.stages[0].configuration_fingerprint
    assert result.trace.stages[0].backend == "local"
    assert observer.snapshot()["documents_processed"] == 1


@dataclass(frozen=True)
class _FailingExtractor:
    def extract(self, source_text: str) -> list[EntityAnnotation]:
        raise RuntimeError(f"failure in {source_text}")


def test_failed_pipeline_exposes_partial_trace_without_raw_text() -> None:
    options = PipelineOptions(
        enable_context=False,
        enable_linking=False,
        enable_candidate_reranking=False,
        enable_entity_kg_validation=False,
        enable_relations=False,
        enable_relation_kg_validation=False,
    )
    runner = PipelineRunner(
        PipelineComponents(entity_extractor=_FailingExtractor(), options=options)
    )
    with pytest.raises(RuntimeError) as raised:
        runner.process_text_with_trace("failed-doc", "private clinical note")
    trace = raised.value.pipeline_trace
    assert trace.stages[-1].status == "failure"
    assert trace.stages[-1].error_type == "RuntimeError"
    assert trace.stages[-1].error_message == "redacted"
    assert "private clinical note" not in str(trace.to_json())


def test_noop_observer_is_default_and_observer_is_thread_safe() -> None:
    options = PipelineOptions(
        enable_context=False,
        enable_linking=False,
        enable_candidate_reranking=False,
        enable_entity_kg_validation=False,
        enable_relations=False,
        enable_relation_kg_validation=False,
    )
    components = PipelineComponents(entity_extractor=_FailingExtractor(), options=options)
    assert isinstance(components.observer, NoOpPipelineObserver)
    observer = InMemoryPipelineObserver()

    def record(index: int) -> None:
        from medical_kg_nlp.pipeline.tracing import PipelineTrace

        trace = PipelineTrace(document_id=f"doc-{index}", observer=observer)
        with trace.stage("test") as counters:
            counters["documents"] = 1
        trace.mark_finished(success=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(record, range(20)))
    assert observer.snapshot()["documents_processed"] == 20
