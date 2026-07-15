from medical_kg_nlp.datasets.synthetic_adapter import SyntheticDatasetAdapter
from medical_kg_nlp.pipeline import PipelineFactory, PipelineFactoryConfig, PipelineOptions
from medical_kg_nlp.schema.types import AssertionStatus


def test_pipeline_trace_records_algorithm_stages() -> None:
    document = SyntheticDatasetAdapter().load_documents("data/samples/sample_notes.jsonl")[0]
    result = PipelineFactory.from_config().process_document_with_trace(document)

    stage_names = [stage.name for stage in result.trace.stages]
    assert stage_names == [
        "document_loader",
        "offset_preserving_preprocessing",
        "section_detection",
        "sentence_splitting",
        "entity_extraction",
        "context_assertion_classification",
        "candidate_generation",
        "candidate_reranking",
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
    assert by_stage["offset_preserving_preprocessing"].counters["offset_map_entries"] > 0
    assert by_stage["offset_preserving_preprocessing"].counters["diagnostic_only"] == 1
    candidate_counters = by_stage["candidate_generation"].counters
    assert candidate_counters["generated_candidates"] + candidate_counters["pinned_entities"] > 0
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
