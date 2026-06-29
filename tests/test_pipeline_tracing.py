from medical_kg_nlp.datasets.synthetic_adapter import SyntheticDatasetAdapter
from medical_kg_nlp.pipeline import PipelineOptions, PipelineRunner
from medical_kg_nlp.schema.types import AssertionStatus


def test_pipeline_trace_records_algorithm_stages() -> None:
    document = SyntheticDatasetAdapter().load_documents("data/samples/sample_notes.jsonl")[0]
    result = PipelineRunner().process_document_with_trace(document)

    stage_names = [stage.name for stage in result.trace.stages]
    assert stage_names == [
        "sentence_split",
        "rule_ner",
        "context_assertion",
        "entity_linking",
        "entity_kg_validation",
        "relation_extraction",
        "relation_kg_validation",
        "prediction_validation",
    ]
    assert result.trace.total_ms >= 0
    assert result.trace.bottleneck() is not None
    assert result.trace.to_json()["bottleneck_stage"] in stage_names
    assert result.prediction.document_id == document.document_id


def test_pipeline_options_can_disable_context_and_relations() -> None:
    document = SyntheticDatasetAdapter().load_documents("data/samples/sample_notes.jsonl")[0]
    runner = PipelineRunner(options=PipelineOptions(enable_context=False, enable_relations=False))
    result = runner.process_document_with_trace(document)

    by_text = {entity.text: entity for entity in result.prediction.entities}
    assert by_text["viêm phổi"].assertion == AssertionStatus.UNKNOWN
    assert result.prediction.relations == []
    relation_stage = next(stage for stage in result.trace.stages if stage.name == "relation_extraction")
    assert relation_stage.counters["skipped_entities"] == len(result.prediction.entities)
