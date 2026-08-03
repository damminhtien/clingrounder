from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from medical_kg_nlp.benchmarks.phase1.phase1 import _match_phase1_rows
from medical_kg_nlp.benchmarks.phase1.ontology import PHASE1_ASSERTABLE_TYPES, PHASE1_CODABLE_TYPES


MANUAL_GOLD_HOLDOUT_MODULUS = 5
MANUAL_GOLD_HOLDOUT_BUCKET = 0
MANUAL_GOLD_SPLIT_SCHEMA_VERSION = "phase1-manual-gold-split.v1"


def manual_gold_split(document_id: str) -> str:
    digest_prefix = hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:8]
    bucket = int(digest_prefix, 16) % MANUAL_GOLD_HOLDOUT_MODULUS
    return "holdout" if bucket == MANUAL_GOLD_HOLDOUT_BUCKET else "train"


def build_manual_gold_split_manifest(
    gold_dir: str | Path,
    documents_dir: str | Path,
) -> dict[str, Any]:
    """Fingerprint the reviewed corpus and freeze its deterministic split.

    The manifest hashes both labels and raw text. A later annotation or source-text edit therefore
    requires an explicit re-freeze instead of silently changing the experiment population.
    """
    gold_root = Path(gold_dir)
    documents_root = Path(documents_dir)
    gold_paths = {
        path.stem: path
        for path in gold_root.glob("*.json")
        if path.stem.isdigit()
    }
    document_paths = {
        path.stem: path
        for path in documents_root.glob("*.txt")
        if path.stem.isdigit()
    }
    if set(gold_paths) != set(document_paths):
        missing_gold = sorted(set(document_paths) - set(gold_paths), key=_document_sort_key)
        missing_documents = sorted(set(gold_paths) - set(document_paths), key=_document_sort_key)
        raise ValueError(
            "Manual-gold/source document mismatch: "
            f"missing_gold={missing_gold}, missing_documents={missing_documents}."
        )

    assignments: list[dict[str, Any]] = []
    gold_entity_count = 0
    for document_id in sorted(gold_paths, key=_document_sort_key):
        payload = json.loads(gold_paths[document_id].read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{gold_paths[document_id]}: expected a JSON list.")
        gold_entity_count += len(payload)
        assignments.append(
            {
                "document_id": document_id,
                "split": manual_gold_split(document_id),
                "gold_sha256": _file_sha256(gold_paths[document_id]),
                "document_sha256": _file_sha256(document_paths[document_id]),
            }
        )

    train_ids = [row["document_id"] for row in assignments if row["split"] == "train"]
    holdout_ids = [row["document_id"] for row in assignments if row["split"] == "holdout"]
    corpus_fingerprint = hashlib.sha256(
        json.dumps(assignments, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": MANUAL_GOLD_SPLIT_SCHEMA_VERSION,
        "split_policy": {
            "algorithm": "int(sha256(document_id)[:8], 16) % 5",
            "holdout_bucket": MANUAL_GOLD_HOLDOUT_BUCKET,
            "modulus": MANUAL_GOLD_HOLDOUT_MODULUS,
        },
        "corpus": {
            "document_count": len(assignments),
            "gold_entity_count": gold_entity_count,
            "fingerprint_sha256": corpus_fingerprint,
        },
        "splits": {
            "train": {
                "document_count": len(train_ids),
                "document_ids": train_ids,
            },
            "holdout": {
                "document_count": len(holdout_ids),
                "document_ids": holdout_ids,
            },
        },
        "assignments": assignments,
    }


def verify_manual_gold_split_manifest(
    manifest: dict[str, Any],
    gold_dir: str | Path,
    documents_dir: str | Path,
) -> None:
    current = build_manual_gold_split_manifest(gold_dir, documents_dir)
    if manifest != current:
        expected = manifest.get("corpus", {}).get("fingerprint_sha256")
        actual = current["corpus"]["fingerprint_sha256"]
        raise ValueError(
            "Frozen manual-gold split no longer matches the corpus: "
            f"expected fingerprint {expected}, got {actual}."
        )


def write_manual_gold_split_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def evaluate_manual_gold(
    gold_by_doc: dict[str, list[dict[str, Any]]],
    pred_by_doc: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    from medical_kg_nlp.benchmarks.phase1.phase1 import score_phase1_documents

    split_ids = {
        "all": sorted(gold_by_doc, key=_document_sort_key),
        "train": sorted((doc_id for doc_id in gold_by_doc if manual_gold_split(doc_id) == "train"), key=_document_sort_key),
        "holdout": sorted(
            (doc_id for doc_id in gold_by_doc if manual_gold_split(doc_id) == "holdout"),
            key=_document_sort_key,
        ),
    }
    split_reports: dict[str, Any] = {}
    all_errors: list[dict[str, Any]] = []
    for split_name, document_ids in split_ids.items():
        split_gold = {doc_id: gold_by_doc[doc_id] for doc_id in document_ids}
        split_pred = {doc_id: pred_by_doc.get(doc_id, []) for doc_id in document_ids}
        metrics, errors = score_phase1_documents(split_gold, split_pred)
        for error in errors:
            error["split"] = manual_gold_split(str(error["document_id"]))
        if split_name == "all":
            all_errors = errors
        split_reports[split_name] = {
            "document_count": len(document_ids),
            "document_ids": document_ids,
            "metrics": metrics,
            "entity_types": _entity_type_metrics(split_gold, split_pred),
            "selective_prediction": _selective_prediction_metrics(split_gold, split_pred),
            "error_counts": dict(sorted(Counter(error["error_type"] for error in errors).items())),
        }
    return {
        "schema_version": "phase1-manual-gold.v1",
        "split_policy": {
            "algorithm": "int(sha256(document_id)[:8], 16) % 5",
            "holdout_bucket": MANUAL_GOLD_HOLDOUT_BUCKET,
        },
        "splits": split_reports,
        "errors": all_errors,
    }


def compare_manual_gold_gate(
    report: dict[str, Any],
    baseline: dict[str, Any],
    *,
    minimum_score_gain: float = 1.0,
    minimum_text_gain: float = 0.015,
    minimum_missing_reduction: int = 20,
    maximum_spurious: int = 28,
    maximum_boundary_increase: int = 5,
) -> dict[str, Any]:
    current = report["splits"]["holdout"]
    baseline_holdout = baseline["local"]["holdout"]
    current_errors = current["error_counts"]
    baseline_errors = baseline_holdout["error_counts"]
    score_gain = float(current["metrics"]["score"]) - float(baseline_holdout["score"])
    text_gain = float(current["metrics"]["text_score"]) - float(baseline_holdout["text_score"])
    missing_reduction = int(baseline_errors["phase1_missing_entity"]) - int(
        current_errors.get("phase1_missing_entity", 0)
    )
    spurious = int(current_errors.get("phase1_spurious_entity", 0))
    boundary_increase = int(current_errors.get("phase1_text_boundary", 0)) - int(
        baseline_errors["phase1_text_boundary"]
    )
    checks = {
        "score_gain": score_gain >= minimum_score_gain,
        "text_gain": text_gain >= minimum_text_gain,
        "missing_reduction": missing_reduction >= minimum_missing_reduction,
        "spurious": spurious <= maximum_spurious,
        "boundary_increase": boundary_increase <= maximum_boundary_increase,
    }
    blocking_checks = {
        "score_gain": checks["score_gain"],
        "text_gain": checks["text_gain"],
    }
    return {
        "passed": all(blocking_checks.values()),
        "checks": checks,
        "blocking_checks": blocking_checks,
        "diagnostic_checks": {
            key: checks[key] for key in ("missing_reduction", "spurious", "boundary_increase")
        },
        "deltas": {
            "score_gain": round(score_gain, 6),
            "text_gain": round(text_gain, 6),
            "missing_reduction": missing_reduction,
            "spurious": spurious,
            "boundary_increase": boundary_increase,
        },
    }


def load_phase1_directory(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    root = Path(path)
    rows: dict[str, list[dict[str, Any]]] = {}
    document_paths = (path for path in root.glob("*.json") if path.stem.isdigit())
    for json_path in sorted(document_paths, key=lambda item: _document_sort_key(item.stem)):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{json_path}: expected a JSON list.")
        rows[json_path.stem] = payload
    return rows


def write_manual_gold_report(report: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    errors = report.get("errors", [])
    with (output / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for row in errors:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    _write_errors_csv(errors, output / "errors.csv")
    (output / "summary.md").write_text(_summary_markdown(report), encoding="utf-8")


def _entity_type_metrics(
    gold_by_doc: dict[str, list[dict[str, Any]]],
    pred_by_doc: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    from medical_kg_nlp.benchmarks.phase1.phase1 import score_phase1_documents

    types = sorted({str(row.get("type", "")) for rows in gold_by_doc.values() for row in rows})
    result: dict[str, Any] = {}
    for entity_type in types:
        gold = {doc_id: [row for row in rows if row.get("type") == entity_type] for doc_id, rows in gold_by_doc.items()}
        pred = {doc_id: [row for row in pred_by_doc.get(doc_id, []) if row.get("type") == entity_type] for doc_id in gold_by_doc}
        metrics, errors = score_phase1_documents(gold, pred)
        result[entity_type] = {
            "metrics": metrics,
            "error_counts": dict(sorted(Counter(error["error_type"] for error in errors).items())),
        }
    return result


def _selective_prediction_metrics(
    gold_by_doc: dict[str, list[dict[str, Any]]],
    pred_by_doc: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    assertion_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    for doc_id, gold_rows in gold_by_doc.items():
        pred_rows = pred_by_doc.get(doc_id, [])
        for match in _match_phase1_rows(gold_rows, pred_rows):
            gold = gold_rows[match.gold_index]
            pred = pred_rows[match.pred_index]
            phase1_type = str(gold.get("type", ""))
            if phase1_type in PHASE1_ASSERTABLE_TYPES:
                _update_selective_counts(assertion_counts, gold.get("assertions"), pred.get("assertions"))
            if phase1_type in PHASE1_CODABLE_TYPES:
                _update_selective_counts(candidate_counts, gold.get("candidates"), pred.get("candidates"))
    return {
        "assertions": _selective_summary(assertion_counts),
        "candidates": _selective_summary(candidate_counts),
    }


def _update_selective_counts(counts: Counter[str], gold_value: Any, pred_value: Any) -> None:
    gold = {str(item) for item in gold_value} if isinstance(gold_value, list) else set()
    pred = {str(item) for item in pred_value} if isinstance(pred_value, list) else set()
    counts["eligible"] += 1
    counts["gold_positive"] += int(bool(gold))
    counts["predicted_positive"] += int(bool(pred))
    counts["true_positive_labels"] += len(gold & pred)
    counts["gold_positive_labels"] += len(gold)
    counts["predicted_positive_labels"] += len(pred)
    counts["null_gold"] += int(not gold)
    counts["null_correct"] += int(not gold and not pred)


def _selective_summary(counts: Counter[str]) -> dict[str, Any]:
    predicted = counts["predicted_positive_labels"]
    gold = counts["gold_positive_labels"]
    eligible = counts["eligible"]
    null_gold = counts["null_gold"]
    return {
        **dict(counts),
        "positive_precision": round(counts["true_positive_labels"] / predicted, 6) if predicted else None,
        "positive_recall": round(counts["true_positive_labels"] / gold, 6) if gold else None,
        "prediction_coverage": round(counts["predicted_positive"] / eligible, 6) if eligible else 0.0,
        "null_accuracy": round(counts["null_correct"] / null_gold, 6) if null_gold else None,
    }


def _write_errors_csv(errors: list[dict[str, Any]], path: Path) -> None:
    fields = ("document_id", "split", "error_type", "span", "text_window", "gold", "prediction")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for error in errors:
            writer.writerow(
                {
                    field: json.dumps(error.get(field), ensure_ascii=False) if field in {"span", "gold", "prediction"} else error.get(field, "")
                    for field in fields
                }
            )


def _summary_markdown(report: dict[str, Any]) -> str:
    lines = ["# Phase 1 Manual-Gold Evaluation", ""]
    for split in ("all", "train", "holdout"):
        row = report["splits"][split]
        metrics = row["metrics"]
        errors = row["error_counts"]
        lines.extend(
            [
                f"## {split.title()}",
                "",
                f"- Documents: {row['document_count']}",
                f"- Score: {metrics['score']}",
                f"- Text score: {metrics['text_score']}",
                f"- Missing: {errors.get('phase1_missing_entity', 0)}",
                f"- Spurious: {errors.get('phase1_spurious_entity', 0)}",
                f"- Boundary: {errors.get('phase1_text_boundary', 0)}",
                "",
            ]
        )
    gate = report.get("gate")
    if isinstance(gate, dict):
        lines.extend(["## Gate", "", f"- Passed: {gate.get('passed')}", ""])
    return "\n".join(lines)


def _document_sort_key(document_id: str) -> tuple[int, int | str]:
    return (0, int(document_id)) if document_id.isdigit() else (1, document_id)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
