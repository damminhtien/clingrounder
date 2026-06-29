from medical_kg_nlp.evaluation.ablation import aggregate_traces, flatten_metrics
from medical_kg_nlp.pipeline.tracing import PipelineTrace, StageMeasurement


def test_aggregate_traces_sums_stage_timings_and_counters() -> None:
    traces = [
        PipelineTrace(
            document_id="d1",
            stages=[
                StageMeasurement("entity_linking", 2.0, {"linked_entities": 3}),
                StageMeasurement("relation_extraction", 1.0, {"relations": 1}),
            ],
        ),
        PipelineTrace(
            document_id="d2",
            stages=[
                StageMeasurement("entity_linking", 4.0, {"linked_entities": 2}),
                StageMeasurement("relation_extraction", 3.0, {"relations": 0}),
            ],
        ),
    ]

    aggregates = aggregate_traces(traces)
    by_stage = {stage.stage: stage for stage in aggregates}

    assert by_stage["entity_linking"].calls == 2
    assert by_stage["entity_linking"].total_ms == 6.0
    assert by_stage["entity_linking"].avg_ms == 3.0
    assert by_stage["entity_linking"].max_ms == 4.0
    assert by_stage["entity_linking"].counters["linked_entities"] == 5
    assert by_stage["relation_extraction"].counters["relations"] == 1


def test_flatten_metrics_keeps_nested_metric_names() -> None:
    flattened = flatten_metrics({"span_exact": {"f1": 0.5}, "context_accuracy": 1.0})
    assert flattened["span_exact.f1"] == 0.5
    assert flattened["context_accuracy"] == 1.0
