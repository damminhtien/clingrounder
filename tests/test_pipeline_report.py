from __future__ import annotations

import copy
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from medical_kg_nlp.datasets.synthetic_adapter import SyntheticDatasetAdapter
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.evaluation.pipeline_report import build_pipeline_report, write_pipeline_report
from medical_kg_nlp.pipeline.tracing import PipelineTrace, StageMeasurement
from medical_kg_nlp.schema.annotation import CandidateConcept
from medical_kg_nlp.schema.document import ClinicalDocument
from medical_kg_nlp.schema.output import ClinicalPrediction
from medical_kg_nlp.schema.types import AssertionStatus, CodeSystem


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
                    name="offset_preserving_preprocessing",
                    elapsed_ms=2.0,
                    counters={
                        "original_characters": 224,
                        "normalized_characters": 224,
                        "offset_map_entries": 224,
                        "diagnostic_only": 1,
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

    report = build_pipeline_report(
        documents=documents,
        gold=gold,
        predictions=predictions,
        traces=traces,
        dictionary=dictionary,
        top_k=5,
    )

    assert report["summary"]["document_count"] == 1
    assert report["runtime"]["bottleneck_stage"] == "candidate_generation"
    assert report["candidate_metrics"]["gold_rank"]["min"] == 2
    assert report["candidate_metrics"]["qualified_candidate_count"]["max"] == 1
    assert report["candidate_metrics"]["qualification_reason_counts"] == {
        "test_qualified": 1,
        "test_rejected": 2,
    }
    assert report["validation"]["summary"]["by_kind"]["invalid_candidate_code_system"] == 1
    assert "score" in report["phase1"]["metrics"]
    assert report["summary"]["phase1_score"] == report["phase1"]["metrics"]["score"]
    assert any(row["stage"] == "phase1_submission" for row in report["stage_metrics"])

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
    assert "Phase 1 score" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_evaluate_pipeline_steps_cli_writes_stage_report(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_pipeline_steps.py",
            "--documents",
            "data/samples/sample_notes.jsonl",
            "--gold",
            "data/samples/gold.jsonl",
            "--pred",
            "data/samples/gold.jsonl",
            "--dictionary",
            "data/dictionaries/seed_concepts.jsonl",
            "--output-dir",
            str(tmp_path),
            "--top-k",
            "5",
        ],
        check=True,
    )

    for filename in [
        "metrics.json",
        "stage_metrics.csv",
        "errors.csv",
        "profile.json",
        "summary.md",
    ]:
        assert (tmp_path / filename).exists()
    traces = json.loads((tmp_path / "traces.json").read_text(encoding="utf-8"))
    assert traces == []
    report = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert "phase1" in report
    assert "phase1_score" in report["summary"]


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
            score=0.9,
            concept_id="ICD10:J18.9",
            source="exact",
            matched_alias="đái tháo đường type 2",
            qualified=True,
            qualification_reason="test_qualified",
        ),
        CandidateConcept(
            code_system=CodeSystem.ICD10,
            code="E11",
            name="Type 2 diabetes mellitus",
            score=0.8,
            concept_id="ICD10:E11",
            source="fuzzy",
            matched_alias="đái tháo đường type 2",
            qualification_reason="test_rejected",
        ),
    ]

    entities["E7"].assertion = AssertionStatus.PRESENT
    entities["E8"].candidates = [
        CandidateConcept(
            code_system=CodeSystem.ICD10,
            code="E11",
            name="Type 2 diabetes mellitus",
            score=0.7,
            concept_id="ICD10:E11",
            source="test",
            matched_alias="metformin",
            qualification_reason="test_rejected",
        )
    ]
