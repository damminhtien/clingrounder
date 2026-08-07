from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

from clingrounder.benchmarks.phase1.pipeline_report import build_phase1_pipeline_report
from clingrounder.datasets.synthetic_adapter import SyntheticDatasetAdapter
from clingrounder.dictionaries.dictionary_store import DictionaryStore
from clingrounder.evaluation.pipeline_report import write_pipeline_report
from clingrounder.pipeline.tracing import PipelineTrace, StageMeasurement
from clingrounder.schema.annotation import CandidateConcept
from clingrounder.schema.document import ClinicalDocument
from clingrounder.schema.output import ClinicalPrediction
from clingrounder.schema.types import AssertionStatus, CodeSystem


def test_pipeline_report_merges_metrics_validation_trace_and_errors(tmp_path: Path) -> None:
    documents, gold = _sample_documents_and_gold()
    predictions = copy.deepcopy(gold)
    _inject_report_errors(predictions[0])
    dictionary = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    traces = [
        PipelineTrace(
            document_id="sample_001",
            stages=[
                StageMeasurement(
                    name="lookup_normalization_diagnostics",
                    elapsed_ms=2.0,
                    counters={
                        "original_characters": 224,
                        "normalized_characters": 224,
                        "offset_map_entries": 224,
                        "source_coordinate_spans": 1,
                        "normalized_text_used_downstream": 0,
                    },
                ),
                StageMeasurement(
                    name="candidate_generation",
                    elapsed_ms=4.0,
                    counters={"generated_candidates": 3, "source_exact": 1},
                ),
            ],
        )
    ]

    report = build_phase1_pipeline_report(
        documents=documents,
        gold=gold,
        predictions=predictions,
        traces=traces,
        dictionary=dictionary,
        top_k=5,
    )

    assert report["summary"]["document_count"] == 1
    assert report["preprocessing_metrics"]["normalized_text_used_downstream"] is False
    assert report["runtime"]["bottleneck_stage"] == "candidate_generation"
    assert report["candidate_metrics"]["gold_rank"]["min"] == 2
    assert report["candidate_metrics"]["qualified_candidate_count"]["max"] == 1
    assert report["candidate_metrics"]["qualification_reason_counts"] == {
        "test_qualified": 1,
        "test_rejected": 2,
    }
    assert report["validation"]["summary"]["by_kind"]["invalid_candidate_code_system"] == 1
    assert "score" in report["task"]["metrics"]
    assert report["summary"]["task_metrics"]["score"] == report["task"]["metrics"]["score"]
    assert any(row["stage"] == "task_phase1" for row in report["stage_metrics"])

    error_counts = Counter(row["error_type"] for row in report["errors"])
    assert error_counts["severe_context_error"] == 1
    assert error_counts["linking_wrong_top1"] == 1
    assert error_counts["invalid_candidate_code_system"] == 1

    write_pipeline_report(report, tmp_path)
    for filename in [
        "metrics.json",
        "stage_metrics.csv",
        "errors.csv",
        "errors.jsonl",
        "profile.json",
        "profile.md",
        "traces.json",
        "summary.md",
    ]:
        assert (tmp_path / filename).exists()

    saved_report = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved_report["summary"]["error_count"] == len(report["errors"])
    assert "phase1 score" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def _sample_documents_and_gold() -> tuple[list[ClinicalDocument], list[ClinicalPrediction]]:
    adapter = SyntheticDatasetAdapter()
    documents = adapter.load_documents("data/samples/sample_notes.jsonl")
    gold = adapter.load_gold("data/samples/gold.jsonl")
    return documents, gold


def _inject_report_errors(prediction: ClinicalPrediction) -> None:
    entities = {entity.id: entity for entity in prediction.entities}
    diabetes = entities["E1"]
    diabetes.code = "J18.9"
    diabetes.candidates = [
        CandidateConcept(
            code_system=CodeSystem.ICD10,
            code="J18.9",
            name="Pneumonia",
            retrieval_score=0.9,
            emit_probability=0.9,
            concept_id="ICD10:J18.9",
            source="exact",
            evidence_sources=("exact",),
            matched_alias="đái tháo đường type 2",
            qualified=True,
            qualification_reason="test_qualified",
        ),
        CandidateConcept(
            code_system=CodeSystem.ICD10,
            code="E11",
            name="Type 2 diabetes mellitus",
            retrieval_score=0.8,
            emit_probability=0.8,
            concept_id="ICD10:E11",
            source="fuzzy",
            evidence_sources=("fuzzy",),
            matched_alias="đái tháo đường type 2",
            qualified=False,
            qualification_reason="test_rejected",
        ),
    ]

    entities["E7"].assertion = AssertionStatus.PRESENT
    entities["E8"].candidates = [
        CandidateConcept(
            code_system=CodeSystem.ICD10,
            code="E11",
            name="Type 2 diabetes mellitus",
            retrieval_score=0.7,
            emit_probability=0.7,
            concept_id="ICD10:E11",
            source="test",
            evidence_sources=("test",),
            matched_alias="metformin",
            qualified=False,
            qualification_reason="test_rejected",
        )
    ]
