"""Phase 1 adapter for the task-neutral stage-wise pipeline report."""

from __future__ import annotations

from typing import Any

from medical_kg_nlp.benchmarks.phase1.phase1 import (
    build_phase1_report,
    phase1_validation_error_rows,
)
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.evaluation.pipeline_report import build_pipeline_report
from medical_kg_nlp.pipeline.tracing import PipelineTrace
from medical_kg_nlp.schema.document import ClinicalDocument
from medical_kg_nlp.schema.output import ClinicalPrediction

__all__ = ["build_phase1_pipeline_report", "build_phase1_task_report"]


def build_phase1_task_report(
    documents: list[ClinicalDocument],
    gold: list[ClinicalPrediction],
    predictions: list[ClinicalPrediction],
    dictionary: DictionaryStore | None,
) -> dict[str, Any]:
    """Adapt Phase 1 scorer output to the generic optional task-report contract."""

    report = build_phase1_report(
        documents=documents,
        gold=gold,
        predictions=predictions,
        dictionary=dictionary,
    )
    report["name"] = "phase1"
    report["validation_error_rows"] = phase1_validation_error_rows(
        _dict_rows(report.get("validation_issues", []))
    )
    return report


def build_phase1_pipeline_report(
    documents: list[ClinicalDocument],
    gold: list[ClinicalPrediction],
    predictions: list[ClinicalPrediction],
    traces: list[PipelineTrace],
    dictionary: DictionaryStore | None,
    *,
    reference_gold: list[ClinicalPrediction] | None = None,
    top_k: int = 20,
) -> dict[str, Any]:
    """Build the generic report plus explicitly injected Phase 1 metrics."""

    task_report = build_phase1_task_report(documents, gold, predictions, dictionary)
    return build_pipeline_report(
        documents,
        gold,
        predictions,
        traces,
        dictionary,
        reference_gold=reference_gold,
        top_k=top_k,
        task_report=task_report,
    )


def _dict_rows(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, dict)]
