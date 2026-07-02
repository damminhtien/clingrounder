from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, cast

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.evaluation.ablation import StageAggregate, aggregate_traces, flatten_metrics
from medical_kg_nlp.evaluation.data_profile import profile_dataset, render_markdown
from medical_kg_nlp.evaluation.end_to_end_metrics import evaluate_predictions
from medical_kg_nlp.evaluation.phase1 import build_phase1_report, phase1_validation_error_rows
from medical_kg_nlp.pipeline.tracing import PipelineTrace
from medical_kg_nlp.preprocessing.section_splitter import split_sections
from medical_kg_nlp.preprocessing.sentence_splitter import split_sentences
from medical_kg_nlp.schema.annotation import EntityAnnotation, RelationAnnotation
from medical_kg_nlp.schema.document import ClinicalDocument, Section, Sentence
from medical_kg_nlp.schema.output import ClinicalPrediction
from medical_kg_nlp.schema.types import AssertionStatus, EntityType
from medical_kg_nlp.schema.validator import PredictionValidator
from medical_kg_nlp.utils.io import write_jsonl
from medical_kg_nlp.utils.text import text_window


REPORT_ERROR_FIELDS = [
    "document_id",
    "stage",
    "error_type",
    "severity",
    "span",
    "text_window",
    "gold",
    "prediction",
    "candidate_rank",
    "candidate_list",
    "validation_path",
    "notes",
]

_SEVERE_CONTEXT_PAIRS = {
    AssertionStatus.NEGATED,
    AssertionStatus.FAMILY,
    AssertionStatus.POSSIBLE,
}


def build_pipeline_report(
    documents: list[ClinicalDocument],
    gold: list[ClinicalPrediction],
    predictions: list[ClinicalPrediction],
    traces: list[PipelineTrace],
    dictionary: DictionaryStore | None,
    *,
    reference_gold: list[ClinicalPrediction] | None = None,
    top_k: int = 20,
) -> dict[str, Any]:
    documents_by_id = {document.document_id: document for document in documents}
    profile = profile_dataset(
        documents=documents,
        gold=gold,
        dictionary=dictionary,
        reference_gold=reference_gold,
        top_k=top_k,
    )
    metrics = evaluate_predictions(gold, predictions)
    validation_issues = _validation_issues(predictions, documents_by_id, dictionary)
    validation_summary = _validation_summary(validation_issues)
    stage_aggregates = aggregate_traces(traces)
    runtime = _runtime_summary(stage_aggregates, len(documents))
    candidate_metrics = _candidate_metrics(gold, predictions)
    section_sentence_metrics = _section_sentence_metrics(documents_by_id, predictions)
    preprocessing_metrics = _preprocessing_metrics(documents_by_id, predictions, stage_aggregates)
    phase1_report = build_phase1_report(
        documents=documents,
        gold=gold,
        predictions=predictions,
        dictionary=dictionary,
    )
    errors = _error_rows(gold, predictions, documents_by_id, validation_issues)
    errors.extend(_dict_list(phase1_report["errors"]))
    errors.extend(phase1_validation_error_rows(_dict_list(phase1_report["validation_issues"])))
    error_summary = dict(Counter(str(row["error_type"]) for row in errors))
    stage_metrics = _stage_metric_rows(
        profile=profile,
        metrics=metrics,
        phase1_report=phase1_report,
        validation_summary=validation_summary,
        runtime=runtime,
        stage_aggregates=stage_aggregates,
        candidate_metrics=candidate_metrics,
        section_sentence_metrics=section_sentence_metrics,
        preprocessing_metrics=preprocessing_metrics,
        error_summary=error_summary,
    )
    summary = _summary(
        documents=documents,
        predictions=predictions,
        metrics=metrics,
        phase1_report=phase1_report,
        validation_summary=validation_summary,
        runtime=runtime,
        errors=errors,
    )

    return {
        "summary": summary,
        "metrics": metrics,
        "phase1": phase1_report,
        "profile": profile,
        "stage_metrics": stage_metrics,
        "validation": {
            "issues": validation_issues,
            "summary": validation_summary,
        },
        "errors": errors,
        "error_summary": error_summary,
        "candidate_metrics": candidate_metrics,
        "section_sentence_metrics": section_sentence_metrics,
        "preprocessing_metrics": preprocessing_metrics,
        "runtime": runtime,
        "traces": [trace.to_json() for trace in traces],
    }


def write_pipeline_report(report: dict[str, Any], output_dir: str | Path) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    _write_json(path / "metrics.json", report)
    _write_stage_metrics_csv(path / "stage_metrics.csv", _dict_list(report["stage_metrics"]))
    _write_errors_csv(path / "errors.csv", _dict_list(report["errors"]))
    write_jsonl(path / "errors.jsonl", _dict_list(report["errors"]))
    _write_json(path / "profile.json", _mapping(report["profile"]))
    (path / "profile.md").write_text(render_markdown(_mapping(report["profile"])), encoding="utf-8")
    _write_json(path / "traces.json", _list(report["traces"]))
    (path / "summary.md").write_text(render_summary_markdown(report), encoding="utf-8")


def render_summary_markdown(report: dict[str, Any]) -> str:
    summary = _mapping(report["summary"])
    runtime = _mapping(report["runtime"])
    validation = _mapping(_mapping(report["validation"])["summary"])
    phase1 = _mapping(report.get("phase1", {}))
    phase1_validation = _mapping(phase1.get("validation_summary", {}))
    error_summary = _mapping(report["error_summary"])
    lines = [
        "# Pipeline Evaluation Summary",
        "",
        "## Run",
        "",
        f"- Documents: {summary['document_count']}",
        f"- Predictions: {summary['prediction_count']}",
        f"- Internal validation issues: {validation['issue_count']}",
        f"- Phase 1 validation issues: {phase1_validation.get('issue_count', 0)}",
        f"- Error rows: {summary['error_count']}",
        f"- Docs/sec: {_format_float(runtime['docs_per_second'])}",
        f"- Bottleneck stage: {runtime['bottleneck_stage'] or 'N/A'}",
        "",
        "## Key Metrics",
        "",
        f"- Phase 1 score: {_format_float(summary.get('phase1_score', 'N/A'))}",
        f"- Phase 1 text: {_format_float(summary.get('phase1_text_score', 'N/A'))}",
        f"- Phase 1 assertions: {_format_float(summary.get('phase1_assertions_score', 'N/A'))}",
        f"- Phase 1 candidates: {_format_float(summary.get('phase1_candidates_score', 'N/A'))}",
        f"- Span exact F1: {_format_float(summary['span_exact_f1'])}",
        f"- Linking accuracy@1: {_format_float(summary['linking_accuracy_at_1'])}",
        f"- Context macro-F1: {_format_float(summary['context_macro_f1'])}",
        f"- Relation F1 (internal only): {_format_float(summary['relation_f1'])}",
        "",
        "## Top Error Types",
        "",
    ]
    if error_summary:
        for error_type, count in sorted(error_summary.items(), key=lambda item: (-int(item[1]), item[0]))[:10]:
            lines.append(f"- {error_type}: {count}")
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def _validation_issues(
    predictions: list[ClinicalPrediction],
    documents_by_id: dict[str, ClinicalDocument],
    dictionary: DictionaryStore | None,
) -> list[dict[str, Any]]:
    validator = PredictionValidator(cast(Any, dictionary))
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        document = documents_by_id.get(prediction.document_id)
        source_text = document.text if document else None
        for issue in validator.validate_prediction(prediction, source_text):
            row: dict[str, Any] = issue.to_json()
            row["document_id"] = prediction.document_id
            rows.append(row)
    return rows


def _validation_summary(issues: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind = Counter(str(issue["kind"]) for issue in issues)
    return {
        "issue_count": len(issues),
        "by_kind": dict(sorted(by_kind.items())),
    }


def _runtime_summary(stage_aggregates: list[StageAggregate], document_count: int) -> dict[str, Any]:
    total_ms = sum(stage.total_ms for stage in stage_aggregates)
    bottleneck = max(stage_aggregates, key=lambda stage: stage.total_ms) if stage_aggregates else None
    return {
        "total_ms": round(total_ms, 6),
        "docs_per_second": round(document_count / (total_ms / 1000), 6) if total_ms > 0 else 0.0,
        "bottleneck_stage": bottleneck.stage if bottleneck else None,
        "stage_aggregates": [stage.to_json() for stage in stage_aggregates],
    }


def _candidate_metrics(
    gold: list[ClinicalPrediction],
    predictions: list[ClinicalPrediction],
) -> dict[str, Any]:
    pred_by_doc = {prediction.document_id: prediction for prediction in predictions}
    candidate_lengths = [
        len(entity.candidates)
        for prediction in predictions
        for entity in prediction.entities
    ]
    source_counts: Counter[str] = Counter()
    margins: list[float] = []
    for prediction in predictions:
        for entity in prediction.entities:
            for candidate in entity.candidates:
                source_counts[candidate.source or "UNKNOWN"] += 1
            if len(entity.candidates) >= 2:
                margins.append(entity.candidates[0].score - entity.candidates[1].score)

    gold_ranks: list[float] = []
    exact_gold_coded = 0
    empty_for_gold = 0
    missing_gold = 0
    for gold_doc in gold:
        pred_doc = pred_by_doc.get(gold_doc.document_id)
        if pred_doc is None:
            continue
        pred_by_key = {_entity_key(entity): entity for entity in pred_doc.entities}
        for gold_entity in gold_doc.entities:
            if gold_entity.code is None:
                continue
            pred_entity = pred_by_key.get(_entity_key(gold_entity))
            if pred_entity is None:
                continue
            exact_gold_coded += 1
            if not pred_entity.candidates:
                empty_for_gold += 1
                continue
            rank = _candidate_rank(pred_entity, gold_entity)
            if rank is None:
                missing_gold += 1
            else:
                gold_ranks.append(float(rank))

    return {
        "candidate_count": _number_summary(candidate_lengths),
        "entities_with_no_candidates": sum(1 for count in candidate_lengths if count == 0),
        "candidate_source_counts": dict(sorted(source_counts.items())),
        "gold_coded_exact_matches": exact_gold_coded,
        "candidate_empty_gold_coded": empty_for_gold,
        "candidate_missing_gold": missing_gold,
        "gold_rank": _number_summary(gold_ranks),
        "top1_margin": _number_summary(margins),
    }


def _section_sentence_metrics(
    documents_by_id: dict[str, ClinicalDocument],
    predictions: list[ClinicalPrediction],
) -> dict[str, Any]:
    outside_section = 0
    outside_sentence = 0
    for prediction in predictions:
        document = documents_by_id.get(prediction.document_id)
        if document is None:
            continue
        sections = split_sections(document.text)
        sentences = _sentences_from_sections(sections, document.text)
        for entity in prediction.entities:
            outside_section += int(not _contained_in_any(entity.span, [section.span for section in sections]))
            outside_sentence += int(not _contained_in_any(entity.span, [sentence.span for sentence in sentences]))
    return {
        "entities_outside_section": outside_section,
        "entities_outside_sentence": outside_sentence,
    }


def _preprocessing_metrics(
    documents_by_id: dict[str, ClinicalDocument],
    predictions: list[ClinicalPrediction],
    stage_aggregates: list[StageAggregate],
) -> dict[str, Any]:
    whitespace_trim_mismatches = 0
    for prediction in predictions:
        document = documents_by_id.get(prediction.document_id)
        if document is None:
            continue
        for entity in prediction.entities:
            start, end = entity.span
            if start < 0 or end < start or end > len(document.text):
                continue
            source_slice = document.text[start:end]
            if source_slice != entity.text and source_slice.strip() == entity.text.strip():
                whitespace_trim_mismatches += 1
    offset_stage = next(
        (stage for stage in stage_aggregates if stage.stage == "offset_preserving_preprocessing"),
        None,
    )
    return {
        "whitespace_trim_mismatch_count": whitespace_trim_mismatches,
        "normalized_text_diagnostic_only": bool(
            offset_stage and offset_stage.counters.get("diagnostic_only", 0) > 0
        ),
    }


def _error_rows(
    gold: list[ClinicalPrediction],
    predictions: list[ClinicalPrediction],
    documents_by_id: dict[str, ClinicalDocument],
    validation_issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(_entity_error_rows(gold, predictions, documents_by_id))
    rows.extend(_relation_error_rows(gold, predictions, documents_by_id))
    rows.extend(_validation_error_rows(validation_issues, documents_by_id))
    return rows


def _entity_error_rows(
    gold: list[ClinicalPrediction],
    predictions: list[ClinicalPrediction],
    documents_by_id: dict[str, ClinicalDocument],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pred_by_doc = {prediction.document_id: prediction for prediction in predictions}
    for gold_doc in gold:
        pred_doc = pred_by_doc.get(gold_doc.document_id)
        document = documents_by_id.get(gold_doc.document_id)
        source_text = document.text if document else ""
        if pred_doc is None:
            rows.append(
                _error_row(
                    document_id=gold_doc.document_id,
                    stage="document_loader",
                    error_type="missing_document",
                    severity="error",
                    notes="No prediction for gold document.",
                )
            )
            continue
        pred_by_key = {_entity_key(entity): entity for entity in pred_doc.entities}
        pred_by_span = _entities_by_span(pred_doc.entities)
        gold_keys = {_entity_key(entity) for entity in gold_doc.entities}
        gold_spans = {entity.span for entity in gold_doc.entities}

        for gold_entity in gold_doc.entities:
            prediction = pred_by_key.get(_entity_key(gold_entity))
            if prediction is None:
                same_span = pred_by_span.get(gold_entity.span)
                overlap = _find_overlap_same_type(gold_entity, pred_doc.entities)
                if same_span is not None:
                    rows.append(
                        _entity_error_row(
                            document_id=gold_doc.document_id,
                            stage="entity_extraction",
                            error_type="type_confusion",
                            severity="error",
                            source_text=source_text,
                            gold=gold_entity,
                            prediction=same_span,
                            notes=f"Gold type {gold_entity.type.value}, predicted {same_span.type.value}.",
                        )
                    )
                elif overlap is not None:
                    rows.append(
                        _entity_error_row(
                            document_id=gold_doc.document_id,
                            stage="entity_extraction",
                            error_type="span_boundary",
                            severity="error",
                            source_text=source_text,
                            gold=gold_entity,
                            prediction=overlap,
                            notes=_boundary_notes(gold_entity, overlap),
                        )
                    )
                else:
                    rows.append(
                        _entity_error_row(
                            document_id=gold_doc.document_id,
                            stage="entity_extraction",
                            error_type="missing_entity",
                            severity="error",
                            source_text=source_text,
                            gold=gold_entity,
                            prediction=None,
                            notes="No exact or relaxed same-type prediction found.",
                        )
                    )
                continue

            rows.extend(_context_error_rows(gold_doc.document_id, source_text, gold_entity, prediction))
            rows.extend(_candidate_error_rows(gold_doc.document_id, source_text, gold_entity, prediction))
            rows.extend(_linking_error_rows(gold_doc.document_id, source_text, gold_entity, prediction))

        for prediction in pred_doc.entities:
            if _entity_key(prediction) in gold_keys:
                continue
            if prediction.span in gold_spans:
                continue
            if _find_overlap_same_type(prediction, gold_doc.entities) is not None:
                continue
            rows.append(
                _entity_error_row(
                    document_id=pred_doc.document_id,
                    stage="entity_extraction",
                    error_type="spurious_entity",
                    severity="error",
                    source_text=source_text,
                    gold=None,
                    prediction=prediction,
                    notes="Prediction has no matching gold entity.",
                )
            )
    return rows


def _context_error_rows(
    document_id: str,
    source_text: str,
    gold: EntityAnnotation,
    prediction: EntityAnnotation,
) -> list[dict[str, Any]]:
    if gold.assertion == prediction.assertion:
        return []
    severe = gold.assertion in _SEVERE_CONTEXT_PAIRS and prediction.assertion == AssertionStatus.PRESENT
    return [
        _entity_error_row(
            document_id=document_id,
            stage="context_assertion_classification",
            error_type="severe_context_error" if severe else "context_confusion",
            severity="error",
            source_text=source_text,
            gold=gold,
            prediction=prediction,
            notes=f"Gold assertion {gold.assertion.value}, predicted {prediction.assertion.value}.",
        )
    ]


def _candidate_error_rows(
    document_id: str,
    source_text: str,
    gold: EntityAnnotation,
    prediction: EntityAnnotation,
) -> list[dict[str, Any]]:
    if gold.code is None:
        return []
    if not prediction.candidates:
        return [
            _entity_error_row(
                document_id=document_id,
                stage="candidate_generation",
                error_type="candidate_empty",
                severity="warning",
                source_text=source_text,
                gold=gold,
                prediction=prediction,
                notes="Gold-coded entity has no generated candidates.",
            )
        ]
    if _candidate_rank(prediction, gold) is None:
        return [
            _entity_error_row(
                document_id=document_id,
                stage="candidate_generation",
                error_type="candidate_missing_gold",
                severity="warning",
                source_text=source_text,
                gold=gold,
                prediction=prediction,
                notes="Gold code is absent from the generated candidate list.",
            )
        ]
    return []


def _linking_error_rows(
    document_id: str,
    source_text: str,
    gold: EntityAnnotation,
    prediction: EntityAnnotation,
) -> list[dict[str, Any]]:
    if gold.code is None:
        return []
    rank = _candidate_rank(prediction, gold)
    if prediction.code is None:
        return [
            _entity_error_row(
                document_id=document_id,
                stage="normalization_assignment",
                error_type="linking_unlinked",
                severity="error",
                source_text=source_text,
                gold=gold,
                prediction=prediction,
                candidate_rank=rank,
                notes="Gold-coded entity remained unlinked.",
            )
        ]
    if prediction.code != gold.code or prediction.code_system != gold.code_system:
        return [
            _entity_error_row(
                document_id=document_id,
                stage="normalization_assignment",
                error_type="linking_wrong_top1",
                severity="error",
                source_text=source_text,
                gold=gold,
                prediction=prediction,
                candidate_rank=rank,
                notes="Assigned code differs from gold code.",
            )
        ]
    return []


def _relation_error_rows(
    gold: list[ClinicalPrediction],
    predictions: list[ClinicalPrediction],
    documents_by_id: dict[str, ClinicalDocument],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pred_by_doc = {prediction.document_id: prediction for prediction in predictions}
    for gold_doc in gold:
        pred_doc = pred_by_doc.get(gold_doc.document_id)
        if pred_doc is None:
            continue
        document = documents_by_id.get(gold_doc.document_id)
        source_text = document.text if document else ""
        pred_keys = {_relation_key(relation) for relation in pred_doc.relations}
        pred_by_endpoints = _relations_by_endpoints(pred_doc.relations)
        gold_keys = {_relation_key(relation) for relation in gold_doc.relations}
        gold_by_endpoints = _relations_by_endpoints(gold_doc.relations)

        for relation in gold_doc.relations:
            if _relation_key(relation) in pred_keys:
                continue
            endpoint_match = pred_by_endpoints.get((relation.head, relation.tail))
            if endpoint_match:
                rows.append(
                    _relation_error_row(
                        document_id=gold_doc.document_id,
                        source_text=source_text,
                        error_type="relation_type_confusion",
                        severity="error",
                        gold=relation,
                        prediction=endpoint_match,
                        notes=(
                            f"Gold relation {relation.type.value}, "
                            f"predicted {endpoint_match.type.value}."
                        ),
                    )
                )
            else:
                rows.append(
                    _relation_error_row(
                        document_id=gold_doc.document_id,
                        source_text=source_text,
                        error_type="missing_relation",
                        severity="error",
                        gold=relation,
                        prediction=None,
                        notes="No prediction for gold relation.",
                    )
                )

        for relation in pred_doc.relations:
            if _relation_key(relation) in gold_keys:
                continue
            if (relation.head, relation.tail) in gold_by_endpoints:
                continue
            rows.append(
                _relation_error_row(
                    document_id=pred_doc.document_id,
                    source_text=source_text,
                    error_type="spurious_relation",
                    severity="error",
                    gold=None,
                    prediction=relation,
                    notes="Predicted relation has no matching gold relation.",
                )
            )
    return rows


def _validation_error_rows(
    validation_issues: list[dict[str, Any]],
    documents_by_id: dict[str, ClinicalDocument],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for issue in validation_issues:
        document_id = str(issue["document_id"])
        document = documents_by_id.get(document_id)
        error_type = _validation_error_type(str(issue["kind"]), str(issue["path"]))
        rows.append(
            _error_row(
                document_id=document_id,
                stage=_validation_stage(error_type),
                error_type=error_type,
                severity="blocking",
                text_window=document.text[:120] if document else "",
                validation_path=str(issue["path"]),
                notes=str(issue["message"]),
            )
        )
    return rows


def _stage_metric_rows(
    *,
    profile: dict[str, Any],
    metrics: dict[str, Any],
    phase1_report: dict[str, Any],
    validation_summary: dict[str, Any],
    runtime: dict[str, Any],
    stage_aggregates: list[StageAggregate],
    candidate_metrics: dict[str, Any],
    section_sentence_metrics: dict[str, Any],
    preprocessing_metrics: dict[str, Any],
    error_summary: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    _add_row(rows, "data_profile", "documents.count", _mapping(profile["documents"])["count"])
    _add_row(rows, "data_profile", "entities.count", _mapping(profile["entities"])["count"])
    _add_row(rows, "data_profile", "relations.count", _mapping(profile["relations"])["count"])
    _add_row(
        rows,
        "data_profile",
        "dictionary_coverage.coverage",
        _mapping(profile["dictionary_coverage"])["coverage"],
    )
    _add_row(rows, "preprocessing_offset", "offsets.issue_count", _mapping(profile["offsets"])["issue_count"])
    _add_mapping(rows, "preprocessing_offset", preprocessing_metrics)
    _add_mapping(rows, "section_sentence", section_sentence_metrics)

    flattened = flatten_metrics(metrics)
    for key, value in flattened.items():
        _add_row(rows, _metric_stage(key), key, value)
    _add_mapping(rows, "phase1_submission", _mapping(phase1_report["metrics"]), "phase1")
    _add_mapping(
        rows,
        "phase1_submission",
        _mapping(phase1_report["validation_summary"]),
        "phase1.validation",
    )
    _add_mapping(rows, "candidate_generation", candidate_metrics)
    _add_mapping(rows, "schema_kg_validation", validation_summary)
    _add_mapping(rows, "error_analysis", error_summary)
    _add_row(rows, "runtime", "total_ms", runtime["total_ms"])
    _add_row(rows, "runtime", "docs_per_second", runtime["docs_per_second"])
    _add_row(rows, "runtime", "bottleneck_stage", runtime["bottleneck_stage"])

    for stage in stage_aggregates:
        _add_row(rows, stage.stage, "runtime.calls", stage.calls)
        _add_row(rows, stage.stage, "runtime.total_ms", round(stage.total_ms, 6))
        _add_row(rows, stage.stage, "runtime.avg_ms", round(stage.avg_ms, 6))
        _add_row(rows, stage.stage, "runtime.max_ms", round(stage.max_ms, 6))
        for key, value in sorted(stage.counters.items()):
            _add_row(rows, stage.stage, f"counter.{key}", value)
    return rows


def _summary(
    *,
    documents: list[ClinicalDocument],
    predictions: list[ClinicalPrediction],
    metrics: dict[str, Any],
    phase1_report: dict[str, Any],
    validation_summary: dict[str, Any],
    runtime: dict[str, Any],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    span_exact = _mapping(metrics["span_exact"])
    relation = _mapping(metrics["relation"])
    phase1_metrics = _mapping(phase1_report["metrics"])
    phase1_validation = _mapping(phase1_report["validation_summary"])
    phase1_issue_count = int(phase1_validation["issue_count"])
    return {
        "document_count": len(documents),
        "prediction_count": len(predictions),
        "validation_issue_count": validation_summary["issue_count"] + phase1_issue_count,
        "internal_validation_issue_count": validation_summary["issue_count"],
        "phase1_validation_issue_count": phase1_issue_count,
        "error_count": len(errors),
        "phase1_score": phase1_metrics["score"],
        "phase1_text_score": phase1_metrics["text_score"],
        "phase1_assertions_score": phase1_metrics["assertions_score"],
        "phase1_candidates_score": phase1_metrics["candidates_score"],
        "span_exact_f1": span_exact["f1"],
        "linking_accuracy_at_1": metrics["linking_accuracy_at_1"],
        "context_macro_f1": metrics["context_macro_f1"],
        "relation_f1": relation["f1"],
        "bottleneck_stage": runtime["bottleneck_stage"],
    }


def _metric_stage(metric_name: str) -> str:
    if metric_name.startswith("span_"):
        return "entity_extraction"
    if metric_name.startswith("linking_"):
        return "normalization_assignment"
    if metric_name.startswith("context_"):
        return "context_assertion_classification"
    if metric_name.startswith("relation"):
        return "relation_extraction"
    return "end_to_end"


def _add_mapping(rows: list[dict[str, Any]], stage: str, payload: dict[str, Any], prefix: str = "") -> None:
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            _add_mapping(rows, stage, value, name)
        elif isinstance(value, bool):
            _add_row(rows, stage, name, int(value))
        elif isinstance(value, (int, float, str)) or value is None:
            _add_row(rows, stage, name, value)
        else:
            _add_row(rows, stage, name, json.dumps(value, ensure_ascii=False, sort_keys=True))


def _add_row(
    rows: list[dict[str, Any]],
    stage: str,
    metric: str,
    value: object,
    notes: str = "",
) -> None:
    rows.append({"stage": stage, "metric": metric, "value": value, "notes": notes})


def _entity_error_row(
    *,
    document_id: str,
    stage: str,
    error_type: str,
    severity: str,
    source_text: str,
    gold: EntityAnnotation | None,
    prediction: EntityAnnotation | None,
    candidate_rank: int | None = None,
    notes: str = "",
) -> dict[str, Any]:
    span = gold.span if gold is not None else prediction.span if prediction is not None else None
    window = text_window(source_text, span, radius=80) if span is not None and source_text else ""
    return _error_row(
        document_id=document_id,
        stage=stage,
        error_type=error_type,
        severity=severity,
        span=list(span) if span is not None else None,
        text_window=window,
        gold=gold.to_json() if gold else None,
        prediction=prediction.to_json() if prediction else None,
        candidate_rank=candidate_rank,
        candidate_list=[candidate.to_json() for candidate in prediction.candidates] if prediction else [],
        notes=notes,
    )


def _relation_error_row(
    *,
    document_id: str,
    source_text: str,
    error_type: str,
    severity: str,
    gold: RelationAnnotation | None,
    prediction: RelationAnnotation | None,
    notes: str,
) -> dict[str, Any]:
    span = gold.evidence_span if gold and gold.evidence_span else prediction.evidence_span if prediction else None
    window = text_window(source_text, span, radius=80) if span is not None and source_text else ""
    return _error_row(
        document_id=document_id,
        stage="relation_extraction",
        error_type=error_type,
        severity=severity,
        span=list(span) if span is not None else None,
        text_window=window,
        gold=gold.to_json() if gold else None,
        prediction=prediction.to_json() if prediction else None,
        notes=notes,
    )


def _error_row(
    *,
    document_id: str,
    stage: str,
    error_type: str,
    severity: str,
    span: list[int] | None = None,
    text_window: str = "",
    gold: dict[str, Any] | None = None,
    prediction: dict[str, Any] | None = None,
    candidate_rank: int | None = None,
    candidate_list: list[dict[str, Any]] | None = None,
    validation_path: str = "",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "stage": stage,
        "error_type": error_type,
        "severity": severity,
        "span": span,
        "text_window": text_window,
        "gold": gold,
        "prediction": prediction,
        "candidate_rank": candidate_rank,
        "candidate_list": candidate_list or [],
        "validation_path": validation_path,
        "notes": notes,
    }


def _entity_key(entity: EntityAnnotation) -> tuple[tuple[int, int], EntityType]:
    return entity.span, entity.type


def _entities_by_span(entities: list[EntityAnnotation]) -> dict[tuple[int, int], EntityAnnotation]:
    by_span: dict[tuple[int, int], EntityAnnotation] = {}
    for entity in entities:
        by_span.setdefault(entity.span, entity)
    return by_span


def _find_overlap_same_type(
    target: EntityAnnotation,
    candidates: list[EntityAnnotation],
) -> EntityAnnotation | None:
    for candidate in candidates:
        if candidate.type != target.type:
            continue
        if candidate.span[0] < target.span[1] and target.span[0] < candidate.span[1]:
            return candidate
    return None


def _boundary_notes(gold: EntityAnnotation, prediction: EntityAnnotation) -> str:
    start_mismatch = gold.span[0] != prediction.span[0]
    end_mismatch = gold.span[1] != prediction.span[1]
    if start_mismatch and end_mismatch:
        return "Start and end offsets differ."
    if start_mismatch:
        return "Start offset differs."
    if end_mismatch:
        return "End offset differs."
    return "Overlapping span differs from gold."


def _candidate_rank(prediction: EntityAnnotation, gold: EntityAnnotation) -> int | None:
    for index, candidate in enumerate(prediction.candidates, start=1):
        if candidate.code_system == gold.code_system and candidate.code == gold.code:
            return index
    return None


def _relation_key(relation: RelationAnnotation) -> tuple[str, str, str]:
    return relation.head, relation.tail, relation.type.value


def _relations_by_endpoints(relations: list[RelationAnnotation]) -> dict[tuple[str, str], RelationAnnotation]:
    by_endpoints: dict[tuple[str, str], RelationAnnotation] = {}
    for relation in relations:
        by_endpoints.setdefault((relation.head, relation.tail), relation)
    return by_endpoints


def _contained_in_any(span: tuple[int, int], containers: list[tuple[int, int]]) -> bool:
    return any(start <= span[0] and span[1] <= end for start, end in containers)


def _sentences_from_sections(sections: list[Section], source_text: str) -> list[Sentence]:
    sentences: list[Sentence] = []
    for section in sections:
        sentences.extend(split_sentences(section.text, section_title=section.title, base_offset=section.span[0]))
    return sentences or [Sentence(span=(0, len(source_text)), text=source_text)]


def _validation_error_type(kind: str, path: str) -> str:
    if kind == "offset":
        return "offset"
    if kind == "invalid_code_system":
        return "invalid_code_system"
    if kind == "invalid_candidate_code_system":
        return "invalid_candidate_code_system"
    if kind == "unknown_dictionary_code":
        return "unknown_dictionary_code"
    if kind in {"invalid_relation", "invalid_evidence_span"}:
        return "invalid_relation"
    if ".candidates[" in path and kind == "schema":
        return "invalid_candidate_code_system"
    return "schema"


def _validation_stage(error_type: str) -> str:
    if error_type == "offset":
        return "preprocessing_offset"
    if error_type in {"invalid_relation"}:
        return "ontology_kg_consistency_check"
    if error_type in {
        "invalid_code_system",
        "unknown_dictionary_code",
        "invalid_candidate_code_system",
    }:
        return "icd_rxnorm_umls_validation"
    return "schema_validation"


def _number_summary(values: list[int] | list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "mean": round(mean(values), 6),
        "median": round(median(values), 6),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_stage_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(path, rows, ["stage", "metric", "value", "notes"])


def _write_errors_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(path, rows, REPORT_ERROR_FIELDS)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _csv_value(value: Any) -> str | int | float | None:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)) or value is None:
        return value
    return str(value)


def _mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("Expected mapping value.")
    return value


def _list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError("Expected list value.")
    return value


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("Expected list of mapping values.")
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise TypeError("Expected list of mapping values.")
        rows.append(item)
    return rows


def _format_float(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.6f}"
    return str(value)
