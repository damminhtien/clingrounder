from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

from medical_kg_nlp.evaluation.manual_gold import manual_gold_split
from medical_kg_nlp.evaluation.phase1 import score_phase1_documents
from medical_kg_nlp.utils.text import normalize_for_match, text_window


PHASE1_ENTITY_TYPE_ORDER = (
    "TRIỆU_CHỨNG",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
    "CHẨN_ĐOÁN",
    "THUỐC",
)


def build_entity_wer_report(
    *,
    gold_by_doc: dict[str, list[dict[str, Any]]],
    pred_by_doc: dict[str, list[dict[str, Any]]],
    documents_by_doc: Mapping[str, str],
    stages: Sequence[tuple[str, dict[str, list[dict[str, Any]]]]] = (),
    annotation_policy: Mapping[str, Any] | None = None,
    public_wer: float | None = None,
    final_source_name: str = "final_only",
) -> dict[str, Any]:
    document_ids = sorted(gold_by_doc, key=_document_sort_key)
    # Submission artifacts contain all 100 documents, while manual gold covers
    # only reviewed documents. Keep every report section on the same reviewed
    # population so unreviewed predictions are not counted as spurious errors.
    reviewed_pred_by_doc = {
        document_id: list(pred_by_doc.get(document_id, [])) for document_id in document_ids
    }
    stage_keys = _stage_entity_keys(stages)
    source_stats: dict[str, Counter[str]] = defaultdict(Counter)
    source_type_stats: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    boundary_rows: list[dict[str, Any]] = []
    error_mentions: Counter[tuple[str, str, str, str]] = Counter()
    error_documents: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    knowledge_stats: dict[str, Counter[str]] = defaultdict(Counter)
    policy_index = _policy_index(annotation_policy or {})

    for document_id in document_ids:
        gold_rows = gold_by_doc.get(document_id, [])
        pred_rows = reviewed_pred_by_doc.get(document_id, [])
        source_text = documents_by_doc.get(document_id, "")
        matches = _match_overlapping_rows(gold_rows, pred_rows)
        matched_gold = {gold_index for gold_index, _ in matches}
        matched_pred = {pred_index for _, pred_index in matches}
        for prediction in pred_rows:
            source = _prediction_source(document_id, prediction, stage_keys, final_source_name)
            entity_type = str(prediction.get("type", ""))
            source_stats[source]["predicted"] += 1
            source_type_stats[(source, entity_type)]["predicted"] += 1
            knowledge = _knowledge_status(prediction, policy_index)
            knowledge_stats[knowledge]["predicted"] += 1
        for gold_index, pred_index in matches:
            gold = gold_rows[gold_index]
            prediction = pred_rows[pred_index]
            source = _prediction_source(document_id, prediction, stage_keys, final_source_name)
            entity_type = str(prediction.get("type", ""))
            knowledge = _knowledge_status(prediction, policy_index)
            source_stats[source]["matched"] += 1
            source_type_stats[(source, entity_type)]["matched"] += 1
            knowledge_stats[knowledge]["matched"] += 1
            if _same_entity_boundary(gold, prediction):
                source_stats[source]["exact_boundary"] += 1
                source_type_stats[(source, entity_type)]["exact_boundary"] += 1
                knowledge_stats[knowledge]["exact_boundary"] += 1
                continue
            source_stats[source]["boundary_error"] += 1
            source_type_stats[(source, entity_type)]["boundary_error"] += 1
            knowledge_stats[knowledge]["boundary_error"] += 1
            row = _boundary_error_row(
                document_id=document_id,
                gold=gold,
                prediction=prediction,
                source=source,
                knowledge_status=knowledge,
                source_text=source_text,
            )
            boundary_rows.append(row)
            _record_error_mention(error_mentions, error_documents, row, "boundary", document_id)
        for index, gold in enumerate(gold_rows):
            if index in matched_gold:
                continue
            knowledge = _knowledge_status(gold, policy_index)
            knowledge_stats[knowledge]["missing"] += 1
            row = {
                "error_type": "missing",
                "entity_type": str(gold.get("type", "")),
                "text": str(gold.get("text", "")),
                "source": "gold_only",
            }
            _record_error_mention(error_mentions, error_documents, row, "missing", document_id)
        for index, prediction in enumerate(pred_rows):
            if index in matched_pred:
                continue
            source = _prediction_source(document_id, prediction, stage_keys, final_source_name)
            entity_type = str(prediction.get("type", ""))
            knowledge = _knowledge_status(prediction, policy_index)
            source_stats[source]["spurious"] += 1
            source_type_stats[(source, entity_type)]["spurious"] += 1
            knowledge_stats[knowledge]["spurious"] += 1
            row = {
                "error_type": "spurious",
                "entity_type": str(prediction.get("type", "")),
                "text": str(prediction.get("text", "")),
                "source": source,
            }
            _record_error_mention(error_mentions, error_documents, row, "spurious", document_id)

    split_summary = _split_summary(gold_by_doc, reviewed_pred_by_doc)
    per_document = _per_document_rows(gold_by_doc, reviewed_pred_by_doc)
    per_type = _per_type_rows(gold_by_doc, reviewed_pred_by_doc)
    per_source = _source_rows(source_stats)
    per_source_type = _source_type_rows(source_type_stats)
    per_knowledge = _knowledge_rows(knowledge_stats)
    boundary_summary = _boundary_summary(boundary_rows)
    top_error_mentions = _top_error_mention_rows(error_mentions, error_documents)
    stage_comparison = _stage_comparison(gold_by_doc, stages, reviewed_pred_by_doc)
    source_ablation = _source_ablation_rows(
        gold_by_doc,
        reviewed_pred_by_doc,
        stage_keys,
        final_source_name,
    )
    all_metrics, all_errors = score_phase1_documents(gold_by_doc, reviewed_pred_by_doc)
    scorer_counts = Counter(row["error_type"] for row in all_errors)
    matched_count = sum(counts["matched"] for counts in source_stats.values())
    missing_count = sum(counts["missing"] for counts in knowledge_stats.values())
    spurious_count = sum(counts["spurious"] for counts in source_stats.values())
    summary = {
        "reviewed_document_count": len(document_ids),
        "gold_entity_count": sum(len(rows) for rows in gold_by_doc.values()),
        "predicted_entity_count": sum(
            len(reviewed_pred_by_doc.get(doc_id, [])) for doc_id in document_ids
        ),
        "matched_entity_count": matched_count,
        "scorer_matched_entity_count": int(all_metrics["matched_entities"]),
        "micro_text_score": float(all_metrics["text_score"]),
        "micro_wer_proxy": _wer_proxy(all_metrics),
        "macro_document_wer_proxy": round(mean(row["wer_proxy"] for row in per_document), 6)
        if per_document
        else 0.0,
        "median_document_wer_proxy": round(median(row["wer_proxy"] for row in per_document), 6)
        if per_document
        else 0.0,
        "public_wer": public_wer,
        "public_local_wer_gap": round(public_wer - _wer_proxy(all_metrics), 6)
        if public_wer is not None
        else None,
        "missing_count": missing_count,
        "spurious_count": spurious_count,
        "boundary_error_count": len(boundary_rows),
        "scorer_missing_count": scorer_counts["phase1_missing_entity"],
        "scorer_spurious_count": scorer_counts["phase1_spurious_entity"],
        "scorer_boundary_error_count": scorer_counts["phase1_text_boundary"],
        "worst_entity_type": min(per_type, key=lambda row: row["text_score"])["entity_type"]
        if per_type
        else None,
        "largest_error_source": max(per_source, key=lambda row: row["total_errors"])["source"]
        if per_source
        else None,
    }
    return {
        "schema_version": "phase1-entity-wer-report.v1",
        "summary": summary,
        "splits": split_summary,
        "per_type": per_type,
        "per_source": per_source,
        "per_source_type": per_source_type,
        "source_ablation": source_ablation,
        "per_knowledge_status": per_knowledge,
        "per_document": per_document,
        "boundary_summary": boundary_summary,
        "boundary_errors": boundary_rows,
        "top_error_mentions": top_error_mentions,
        "stage_comparison": stage_comparison,
    }


def write_entity_wer_report(report: Mapping[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "metrics.json", report)
    _write_csv(output / "per_type.csv", _dict_list(report.get("per_type")))
    _write_csv(output / "per_source.csv", _dict_list(report.get("per_source")))
    _write_csv(output / "per_source_type.csv", _dict_list(report.get("per_source_type")))
    _write_csv(output / "source_ablation.csv", _dict_list(report.get("source_ablation")))
    _write_csv(output / "per_knowledge_status.csv", _dict_list(report.get("per_knowledge_status")))
    _write_csv(output / "per_document.csv", _dict_list(report.get("per_document")))
    _write_csv(output / "boundary_errors.csv", _dict_list(report.get("boundary_errors")))
    _write_csv(output / "top_error_mentions.csv", _dict_list(report.get("top_error_mentions")))
    _write_csv(output / "stage_comparison.csv", _dict_list(report.get("stage_comparison")))
    (output / "summary.md").write_text(render_entity_wer_markdown(report), encoding="utf-8")


def render_entity_wer_markdown(report: Mapping[str, Any]) -> str:
    summary = _mapping(report.get("summary"))
    lines = [
        "# Phase 1 Entity WER / Source / Boundary Report",
        "",
        "## Summary",
        "",
        f"- Reviewed documents: {summary.get('reviewed_document_count', 0)}",
        f"- Gold entities: {summary.get('gold_entity_count', 0)}",
        f"- Predicted entities: {summary.get('predicted_entity_count', 0)}",
        f"- Micro WER proxy: {_format(summary.get('micro_wer_proxy'))}",
        f"- Macro-document WER proxy: {_format(summary.get('macro_document_wer_proxy'))}",
        f"- Public WER: {_format(summary.get('public_wer'))}",
        f"- Public/local WER gap: {_format(summary.get('public_local_wer_gap'))}",
        f"- Missing / spurious / boundary: {summary.get('missing_count', 0)} / {summary.get('spurious_count', 0)} / {summary.get('boundary_error_count', 0)}",
        f"- Text-scorer missing / spurious / boundary: {summary.get('scorer_missing_count', 0)} / {summary.get('scorer_spurious_count', 0)} / {summary.get('scorer_boundary_error_count', 0)}",
        f"- Worst entity type: `{summary.get('worst_entity_type', '')}`",
        "- WER uses the Phase 1 text scorer; error taxonomy uses same-type raw-span overlap alignment.",
        "",
        "## Entity Types",
        "",
        "| Type | Gold | Pred | Text score | WER proxy | Missing | Spurious | Boundary |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _dict_list(report.get("per_type")):
        lines.append(
            "| {entity_type} | {gold_count} | {predicted_count} | {text_score:.4f} | "
            "{wer_proxy:.4f} | {missing_count} | {spurious_count} | {boundary_error_count} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Sources",
            "",
            "| Source | Pred | Matched | Exact boundary | Boundary | Spurious | Match rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in _dict_list(report.get("per_source")):
        lines.append(
            "| {source} | {predicted} | {matched} | {exact_boundary} | {boundary_error} | "
            "{spurious} | {match_rate:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Source Ablation",
            "",
            "Positive delta means removing the source makes WER worse, so the source is useful.",
            "",
            "| Source removed | Entities | WER without | Delta WER | Effect |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in _dict_list(report.get("source_ablation")):
        lines.append(
            "| {source} | {removed_count} | {wer_without_source:.4f} | "
            "{delta_wer_without_source:.4f} | {source_effect} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Boundary Kinds",
            "",
            "| Type | Kind | Count |",
            "| --- | --- | ---: |",
        ]
    )
    boundary_summary = _mapping(report.get("boundary_summary"))
    for row in _dict_list(boundary_summary.get("by_type_and_kind")):
        lines.append(f"| {row['entity_type']} | `{row['boundary_kind']}` | {row['count']} |")
    lines.extend(
        [
            "",
            "## Top Boundary Fragments",
            "",
            "| Type | Fragment side | Fragment | Count |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for row in _dict_list(boundary_summary.get("top_fragments"))[:20]:
        lines.append(
            f"| {row['entity_type']} | `{row['fragment_kind']}` | {row['fragment']} | {row['count']} |"
        )
    lines.extend(
        [
            "",
            "## Stage Comparison",
            "",
            "| Stage | Pred | WER proxy | Delta WER | Missing | Spurious | Boundary |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in _dict_list(report.get("stage_comparison")):
        lines.append(
            "| {stage} | {predicted_count} | {wer_proxy:.4f} | {delta_wer_proxy:.4f} | "
            "{missing_count} | {spurious_count} | {boundary_error_count} |".format(**row)
        )
    lines.append("")
    return "\n".join(lines)


def _match_overlapping_rows(
    gold_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    """Align same-type entities only when their raw spans overlap.

    The Phase 1 text scorer intentionally allows same-text mentions at distant
    offsets to align. That is useful for its WER proxy, but unsuitable for
    diagnosing NER boundaries because repeated mentions can be paired across a
    document. This alignment is therefore report-local and offset constrained.
    """

    candidates: list[tuple[tuple[int, float, float, int, int], int, int]] = []
    for gold_index, gold in enumerate(gold_rows):
        gold_start, gold_end = _position(gold)
        gold_length = gold_end - gold_start
        if gold_length <= 0:
            continue
        for pred_index, prediction in enumerate(pred_rows):
            if gold.get("type") != prediction.get("type"):
                continue
            pred_start, pred_end = _position(prediction)
            pred_length = pred_end - pred_start
            if pred_length <= 0:
                continue
            overlap = min(gold_end, pred_end) - max(gold_start, pred_start)
            if overlap <= 0:
                continue
            union = max(gold_end, pred_end) - min(gold_start, pred_start)
            exact_span = int((gold_start, gold_end) == (pred_start, pred_end))
            same_text = int(str(gold.get("text", "")) == str(prediction.get("text", "")))
            boundary_distance = abs(pred_start - gold_start) + abs(pred_end - gold_end)
            quality = (
                exact_span,
                overlap / min(gold_length, pred_length),
                overlap / union,
                same_text,
                -boundary_distance,
            )
            candidates.append((quality, gold_index, pred_index))

    matches: list[tuple[int, int]] = []
    matched_gold: set[int] = set()
    matched_pred: set[int] = set()
    for _, gold_index, pred_index in sorted(
        candidates,
        key=lambda row: (row[0], -row[1], -row[2]),
        reverse=True,
    ):
        if gold_index in matched_gold or pred_index in matched_pred:
            continue
        matched_gold.add(gold_index)
        matched_pred.add(pred_index)
        matches.append((gold_index, pred_index))
    return matches


def _alignment_counts(
    gold_by_doc: Mapping[str, list[dict[str, Any]]],
    pred_by_doc: Mapping[str, list[dict[str, Any]]],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for document_id, gold_rows in gold_by_doc.items():
        pred_rows = pred_by_doc.get(document_id, [])
        matches = _match_overlapping_rows(gold_rows, pred_rows)
        counts["matched"] += len(matches)
        counts["missing"] += len(gold_rows) - len(matches)
        counts["spurious"] += len(pred_rows) - len(matches)
        counts["boundary"] += sum(
            not _same_entity_boundary(gold_rows[gold_index], pred_rows[pred_index])
            for gold_index, pred_index in matches
        )
    return counts


def _split_summary(
    gold_by_doc: dict[str, list[dict[str, Any]]],
    pred_by_doc: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    ids = {
        "all": sorted(gold_by_doc, key=_document_sort_key),
        "train": sorted(
            (doc_id for doc_id in gold_by_doc if manual_gold_split(doc_id) == "train"),
            key=_document_sort_key,
        ),
        "holdout": sorted(
            (doc_id for doc_id in gold_by_doc if manual_gold_split(doc_id) == "holdout"),
            key=_document_sort_key,
        ),
    }
    result: dict[str, Any] = {}
    for split, document_ids in ids.items():
        gold = {doc_id: gold_by_doc[doc_id] for doc_id in document_ids}
        pred = {doc_id: pred_by_doc.get(doc_id, []) for doc_id in document_ids}
        metrics, errors = score_phase1_documents(gold, pred)
        alignment = _alignment_counts(gold, pred)
        document_wer = [
            _single_document_wer(doc_id, gold_by_doc, pred_by_doc) for doc_id in document_ids
        ]
        result[split] = {
            "document_count": len(document_ids),
            "text_score": metrics["text_score"],
            "micro_wer_proxy": _wer_proxy(metrics),
            "macro_document_wer_proxy": round(mean(document_wer), 6) if document_wer else 0.0,
            "median_document_wer_proxy": round(median(document_wer), 6) if document_wer else 0.0,
            "alignment_error_counts": {
                "missing": alignment["missing"],
                "spurious": alignment["spurious"],
                "boundary": alignment["boundary"],
            },
            "scorer_error_counts": dict(
                sorted(Counter(row["error_type"] for row in errors).items())
            ),
        }
    return result


def _per_document_rows(
    gold_by_doc: dict[str, list[dict[str, Any]]],
    pred_by_doc: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document_id in sorted(gold_by_doc, key=_document_sort_key):
        metrics, errors = score_phase1_documents(
            {document_id: gold_by_doc[document_id]},
            {document_id: pred_by_doc.get(document_id, [])},
        )
        counts = Counter(row["error_type"] for row in errors)
        alignment = _alignment_counts(
            {document_id: gold_by_doc[document_id]},
            {document_id: pred_by_doc.get(document_id, [])},
        )
        rows.append(
            {
                "document_id": document_id,
                "split": manual_gold_split(document_id),
                "gold_count": len(gold_by_doc[document_id]),
                "predicted_count": len(pred_by_doc.get(document_id, [])),
                "matched_count": alignment["matched"],
                "scorer_matched_count": metrics["matched_entities"],
                "text_score": metrics["text_score"],
                "wer_proxy": _wer_proxy(metrics),
                "missing_count": alignment["missing"],
                "spurious_count": alignment["spurious"],
                "boundary_error_count": alignment["boundary"],
                "scorer_missing_count": counts["phase1_missing_entity"],
                "scorer_spurious_count": counts["phase1_spurious_entity"],
                "scorer_boundary_error_count": counts["phase1_text_boundary"],
            }
        )
    return sorted(
        rows,
        key=lambda row: (-float(row["wer_proxy"]), _document_sort_key(str(row["document_id"]))),
    )


def _per_type_rows(
    gold_by_doc: dict[str, list[dict[str, Any]]],
    pred_by_doc: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entity_type in PHASE1_ENTITY_TYPE_ORDER:
        gold = {
            doc_id: [row for row in document_rows if row.get("type") == entity_type]
            for doc_id, document_rows in gold_by_doc.items()
        }
        pred = {
            doc_id: [row for row in pred_by_doc.get(doc_id, []) if row.get("type") == entity_type]
            for doc_id in gold_by_doc
        }
        metrics, errors = score_phase1_documents(gold, pred)
        counts = Counter(row["error_type"] for row in errors)
        alignment = _alignment_counts(gold, pred)
        gold_lengths = [
            len(str(row.get("text", "")))
            for document_rows in gold.values()
            for row in document_rows
        ]
        pred_lengths = [
            len(str(row.get("text", "")))
            for document_rows in pred.values()
            for row in document_rows
        ]
        rows.append(
            {
                "entity_type": entity_type,
                "gold_count": metrics["gold_entities"],
                "predicted_count": metrics["predicted_entities"],
                "matched_count": alignment["matched"],
                "scorer_matched_count": metrics["matched_entities"],
                "text_score": float(metrics["text_score"]),
                "wer_proxy": _wer_proxy(metrics),
                "missing_count": alignment["missing"],
                "spurious_count": alignment["spurious"],
                "boundary_error_count": alignment["boundary"],
                "scorer_missing_count": counts["phase1_missing_entity"],
                "scorer_spurious_count": counts["phase1_spurious_entity"],
                "scorer_boundary_error_count": counts["phase1_text_boundary"],
                "gold_span_mean": _mean_length(gold_lengths),
                "gold_span_median": _median_length(gold_lengths),
                "pred_span_mean": _mean_length(pred_lengths),
                "pred_span_median": _median_length(pred_lengths),
            }
        )
    return rows


def _stage_comparison(
    gold_by_doc: dict[str, list[dict[str, Any]]],
    stages: Sequence[tuple[str, dict[str, list[dict[str, Any]]]]],
    final_predictions: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_wer: float | None = None
    for stage, predictions in [*stages, ("final", final_predictions)]:
        pred = {doc_id: predictions.get(doc_id, []) for doc_id in gold_by_doc}
        metrics, errors = score_phase1_documents(gold_by_doc, pred)
        counts = Counter(row["error_type"] for row in errors)
        alignment = _alignment_counts(gold_by_doc, pred)
        wer = _wer_proxy(metrics)
        rows.append(
            {
                "stage": stage,
                "predicted_count": metrics["predicted_entities"],
                "matched_count": alignment["matched"],
                "scorer_matched_count": metrics["matched_entities"],
                "text_score": metrics["text_score"],
                "wer_proxy": wer,
                "delta_wer_proxy": round(wer - previous_wer, 6)
                if previous_wer is not None
                else 0.0,
                "missing_count": alignment["missing"],
                "spurious_count": alignment["spurious"],
                "boundary_error_count": alignment["boundary"],
                "scorer_missing_count": counts["phase1_missing_entity"],
                "scorer_spurious_count": counts["phase1_spurious_entity"],
                "scorer_boundary_error_count": counts["phase1_text_boundary"],
            }
        )
        previous_wer = wer
    return rows


def _source_ablation_rows(
    gold_by_doc: dict[str, list[dict[str, Any]]],
    final_predictions: dict[str, list[dict[str, Any]]],
    stage_keys: Sequence[tuple[str, Mapping[str, set[tuple[str, str, int, int]]]]],
    final_source_name: str,
) -> list[dict[str, Any]]:
    baseline_metrics, _ = score_phase1_documents(gold_by_doc, final_predictions)
    baseline_wer = _wer_proxy(baseline_metrics)
    sources = sorted(
        {
            _prediction_source(document_id, row, stage_keys, final_source_name)
            for document_id, rows in final_predictions.items()
            for row in rows
        }
    )
    result: list[dict[str, Any]] = []
    for source in sources:
        without_source = {
            document_id: [
                row
                for row in rows
                if _prediction_source(document_id, row, stage_keys, final_source_name) != source
            ]
            for document_id, rows in final_predictions.items()
        }
        metrics, _ = score_phase1_documents(gold_by_doc, without_source)
        alignment = _alignment_counts(gold_by_doc, without_source)
        without_wer = _wer_proxy(metrics)
        removed_count = sum(
            len(final_predictions.get(document_id, [])) - len(without_source.get(document_id, []))
            for document_id in gold_by_doc
        )
        delta = round(without_wer - baseline_wer, 6)
        result.append(
            {
                "source": source,
                "removed_count": removed_count,
                "baseline_wer_proxy": baseline_wer,
                "wer_without_source": without_wer,
                "delta_wer_without_source": delta,
                "source_effect": "helpful"
                if delta > 0.0
                else "harmful"
                if delta < 0.0
                else "neutral",
                "missing_without_source": alignment["missing"],
                "spurious_without_source": alignment["spurious"],
                "boundary_without_source": alignment["boundary"],
            }
        )
    return sorted(
        result, key=lambda row: (-float(row["delta_wer_without_source"]), str(row["source"]))
    )


def _boundary_error_row(
    *,
    document_id: str,
    gold: Mapping[str, Any],
    prediction: Mapping[str, Any],
    source: str,
    knowledge_status: str,
    source_text: str,
) -> dict[str, Any]:
    gold_start, gold_end = _position(gold)
    pred_start, pred_end = _position(prediction)
    span = (gold_start, gold_end)
    return {
        "document_id": document_id,
        "split": manual_gold_split(document_id),
        "entity_type": str(gold.get("type", "")),
        "source": source,
        "knowledge_status": knowledge_status,
        "boundary_kind": _boundary_kind(gold_start, gold_end, pred_start, pred_end),
        "gold_text": str(gold.get("text", "")),
        "prediction_text": str(prediction.get("text", "")),
        "gold_start": gold_start,
        "gold_end": gold_end,
        "prediction_start": pred_start,
        "prediction_end": pred_end,
        "start_delta": pred_start - gold_start,
        "end_delta": pred_end - gold_end,
        "missing_prefix": source_text[gold_start:pred_start].strip()
        if pred_start > gold_start
        else "",
        "extra_prefix": source_text[pred_start:gold_start].strip()
        if pred_start < gold_start
        else "",
        "missing_suffix": source_text[pred_end:gold_end].strip() if pred_end < gold_end else "",
        "extra_suffix": source_text[gold_end:pred_end].strip() if pred_end > gold_end else "",
        "text_window": text_window(source_text, span, radius=60)
        if source_text
        else str(gold.get("text", "")),
    }


def _boundary_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    kind_counts = Counter((row["entity_type"], row["boundary_kind"]) for row in rows)
    fragment_counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        for field in ("missing_prefix", "extra_prefix", "missing_suffix", "extra_suffix"):
            fragment = normalize_for_match(str(row[field]))
            if fragment:
                fragment_counts[(str(row["entity_type"]), field, fragment)] += 1
    return {
        "count": len(rows),
        "by_kind": dict(sorted(Counter(row["boundary_kind"] for row in rows).items())),
        "by_type_and_kind": [
            {"entity_type": entity_type, "boundary_kind": kind, "count": count}
            for (entity_type, kind), count in sorted(kind_counts.items())
        ],
        "top_fragments": [
            {
                "entity_type": entity_type,
                "fragment_kind": kind,
                "fragment": fragment,
                "count": count,
            }
            for (entity_type, kind, fragment), count in fragment_counts.most_common(100)
        ],
    }


def _source_rows(stats: Mapping[str, Counter[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, counts in sorted(stats.items()):
        predicted = counts["predicted"]
        matched = counts["matched"]
        rows.append(
            {
                "source": source,
                "predicted": predicted,
                "matched": matched,
                "exact_boundary": counts["exact_boundary"],
                "boundary_error": counts["boundary_error"],
                "spurious": counts["spurious"],
                "total_errors": counts["boundary_error"] + counts["spurious"],
                "match_rate": round(matched / predicted, 6) if predicted else 0.0,
                "exact_boundary_rate": round(counts["exact_boundary"] / matched, 6)
                if matched
                else 0.0,
            }
        )
    return sorted(rows, key=lambda row: (-int(row["total_errors"]), str(row["source"])))


def _source_type_rows(stats: Mapping[tuple[str, str], Counter[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (source, entity_type), counts in sorted(stats.items()):
        predicted = counts["predicted"]
        matched = counts["matched"]
        rows.append(
            {
                "source": source,
                "entity_type": entity_type,
                "predicted": predicted,
                "matched": matched,
                "exact_boundary": counts["exact_boundary"],
                "boundary_error": counts["boundary_error"],
                "spurious": counts["spurious"],
                "match_rate": round(matched / predicted, 6) if predicted else 0.0,
                "exact_boundary_rate": round(counts["exact_boundary"] / matched, 6)
                if matched
                else 0.0,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row["source"]),
            PHASE1_ENTITY_TYPE_ORDER.index(str(row["entity_type"]))
            if row["entity_type"] in PHASE1_ENTITY_TYPE_ORDER
            else len(PHASE1_ENTITY_TYPE_ORDER),
        ),
    )


def _knowledge_rows(stats: Mapping[str, Counter[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for status, counts in sorted(stats.items()):
        predicted = counts["predicted"]
        matched = counts["matched"]
        rows.append(
            {
                "knowledge_status": status,
                "predicted": predicted,
                "matched": matched,
                "exact_boundary": counts["exact_boundary"],
                "boundary_error": counts["boundary_error"],
                "spurious": counts["spurious"],
                "missing": counts["missing"],
                "match_rate": round(matched / predicted, 6) if predicted else 0.0,
            }
        )
    return sorted(rows, key=lambda row: (-int(row["spurious"]), str(row["knowledge_status"])))


def _top_error_mention_rows(
    counts: Counter[tuple[str, str, str, str]],
    documents: Mapping[tuple[str, str, str, str], set[str]],
) -> list[dict[str, Any]]:
    return [
        {
            "error_type": key[0],
            "entity_type": key[1],
            "normalized_text": key[2],
            "source": key[3],
            "count": count,
            "document_count": len(documents[key]),
            "document_ids": ",".join(sorted(documents[key], key=_document_sort_key)),
        }
        for key, count in counts.most_common(200)
    ]


def _record_error_mention(
    counts: Counter[tuple[str, str, str, str]],
    documents: dict[tuple[str, str, str, str], set[str]],
    row: Mapping[str, Any],
    error_type: str,
    document_id: str,
) -> None:
    text = str(row.get("gold_text") or row.get("text") or "")
    key = (
        error_type,
        str(row.get("entity_type", "")),
        normalize_for_match(text),
        str(row.get("source", "")),
    )
    counts[key] += 1
    documents[key].add(document_id)


def _stage_entity_keys(
    stages: Sequence[tuple[str, dict[str, list[dict[str, Any]]]]],
) -> list[tuple[str, dict[str, set[tuple[str, str, int, int]]]]]:
    result: list[tuple[str, dict[str, set[tuple[str, str, int, int]]]]] = []
    for stage, rows_by_doc in stages:
        result.append(
            (
                stage,
                {
                    document_id: {_entity_key(row) for row in rows}
                    for document_id, rows in rows_by_doc.items()
                },
            )
        )
    return result


def _prediction_source(
    document_id: str,
    row: Mapping[str, Any],
    stage_keys: Sequence[tuple[str, Mapping[str, set[tuple[str, str, int, int]]]]],
    fallback: str,
) -> str:
    key = _entity_key(row)
    for stage, keys_by_doc in stage_keys:
        if key in keys_by_doc.get(document_id, set()):
            return stage
    return fallback


def _policy_index(policy: Mapping[str, Any]) -> dict[str, dict[str, set[str]] | set[str]]:
    aliases = _mapping(policy.get("aliases"))
    index: dict[str, dict[str, set[str]] | set[str]] = {}
    for status in ("strict", "context_required", "reviewed"):
        values = _mapping(aliases.get(status))
        index[status] = {
            entity_type: set(_string_list(items)) for entity_type, items in values.items()
        }
    index["unstable"] = set(_string_list(policy.get("unstable_mentions")))
    strict_exclusions = _mapping(_mapping(policy.get("exclusions")).get("strict"))
    index["excluded"] = {
        normalized for values in strict_exclusions.values() for normalized in _string_list(values)
    }
    return index


def _knowledge_status(row: Mapping[str, Any], index: Mapping[str, Any]) -> str:
    normalized = normalize_for_match(str(row.get("text", "")))
    entity_type = str(row.get("type", ""))
    if normalized in set(index.get("unstable", set())):
        return "unstable"
    for status in ("strict", "context_required", "reviewed"):
        by_type = index.get(status, {})
        if isinstance(by_type, Mapping) and normalized in set(by_type.get(entity_type, set())):
            return status
    if normalized in set(index.get("excluded", set())):
        return "strict_exclusion"
    return "unknown"


def _boundary_kind(gold_start: int, gold_end: int, pred_start: int, pred_end: int) -> str:
    if gold_start == pred_start and pred_end < gold_end:
        return "end_under"
    if gold_start == pred_start and pred_end > gold_end:
        return "end_over"
    if gold_end == pred_end and pred_start > gold_start:
        return "start_under"
    if gold_end == pred_end and pred_start < gold_start:
        return "start_over"
    if gold_start <= pred_start and pred_end <= gold_end:
        return "under_both"
    if pred_start <= gold_start and gold_end <= pred_end:
        return "over_both"
    return "crossing"


def _same_entity_boundary(gold: Mapping[str, Any], prediction: Mapping[str, Any]) -> bool:
    return gold.get("position") == prediction.get("position") and gold.get(
        "text"
    ) == prediction.get("text")


def _position(row: Mapping[str, Any]) -> tuple[int, int]:
    value = row.get("position")
    if not isinstance(value, list) or len(value) != 2:
        return (-1, -1)
    return int(value[0]), int(value[1])


def _entity_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    start, end = _position(row)
    return str(row.get("type", "")), str(row.get("text", "")), start, end


def _single_document_wer(
    document_id: str,
    gold_by_doc: dict[str, list[dict[str, Any]]],
    pred_by_doc: dict[str, list[dict[str, Any]]],
) -> float:
    metrics, _ = score_phase1_documents(
        {document_id: gold_by_doc[document_id]},
        {document_id: pred_by_doc.get(document_id, [])},
    )
    return _wer_proxy(metrics)


def _wer_proxy(metrics: Mapping[str, Any]) -> float:
    return round(100.0 * (1.0 - float(metrics["text_score"])), 6)


def _mean_length(values: list[int]) -> float:
    return round(mean(values), 6) if values else 0.0


def _median_length(values: list[int]) -> float:
    return round(median(values), 6) if values else 0.0


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list | tuple | set) else []


def _document_sort_key(document_id: str) -> tuple[int, int | str]:
    return (0, int(document_id)) if document_id.isdigit() else (1, document_id)
