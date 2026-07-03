from __future__ import annotations

import hashlib
import json
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.evaluation.phase1 import validate_phase1_submission_zip

_CODABLE_TYPES = {"CHẨN_ĐOÁN", "THUỐC"}
_ASSERTABLE_TYPES = {"CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"}
_DOSE_LIKE_RE = re.compile(r"(?i)^\d+(?:[\.,]\d+)?\s?(?:mg|g|mcg|ml)$")
_HISTORICAL_MARKERS = (
    "thuốc trước khi nhập viện",
    "trước khi nhập viện",
    "tiền sử bệnh",
    "bệnh nền",
    "bệnh lý mạn tính",
    "bệnh lý mãn tính",
    "đã sử dụng",
    "đang dùng",
    "dùng tại nhà",
)
_HISTORICAL_FALSE_POSITIVE_MARKERS = ("tiền sử bệnh hiện tại", "bệnh sử hiện tại", "lịch sử bệnh hiện tại")
_NEGATION_MARKERS = ("không ghi nhận", "không có", "không thấy", "phủ nhận", "âm tính", "không còn")
_POSSIBLE_MARKERS = ("không loại trừ", "nghi", "theo dõi", "có thể")
_COVERAGE_TERMS = {
    "symptom": (
        "nôn",
        "khó thở",
        "yếu",
        "sốt",
        "đau bụng",
        "đau ngực",
        "buồn nôn",
        "ho",
        "phù",
        "mệt mỏi",
        "đánh trống ngực",
        "thắt chặt ngực",
        "ngất",
        "chóng mặt",
        "khó chịu vùng ngực",
    ),
    "disease": (
        "tăng huyết áp",
        "ung thư",
        "đái tháo đường",
        "nhiễm trùng",
        "nhiễm khuẩn",
        "rung nhĩ",
        "sỏi",
        "bệnh tim mạch",
        "rối loạn lipid máu",
        "thiếu máu",
    ),
    "drug": (
        "Tylenol",
        "Lasix",
        "Omeprazole",
        "Metoprolol",
        "Nitroglycerin",
        "Vancomycin",
        "Prednisone",
        "Albuterol",
        "Aspirin",
        "Doxycycline",
        "atenolol",
    ),
    "lab": ("creatinine", "kali", "bạch cầu", "bilirubin", "hct", "troponin", "inr", "alt", "ast", "glucose"),
}


def build_phase1_submission_analysis(
    *,
    input_dir: str | Path,
    zip_path: str | Path,
    dictionary: DictionaryStore,
    expected_count: int = 100,
    external_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    input_path = Path(input_dir)
    archive_path = Path(zip_path)
    texts = _load_texts(input_path)
    rows_by_document = _load_zip_rows(archive_path)
    validation_issues = [
        issue.to_json()
        for issue in validate_phase1_submission_zip(
            archive_path,
            input_dir=input_path,
            dictionary=dictionary,
            expected_count=expected_count,
        )
    ]
    profile, all_items = _profile_rows(rows_by_document)
    errors, heuristic_counts = _heuristic_errors(texts, all_items, profile)
    phase1_metrics = _phase1_metrics(external_metrics)
    error_summary = _error_summary(profile, heuristic_counts, phase1_metrics)
    if not errors:
        errors = []
    report = {
        "summary": {
            "run_id": "phase1_pre_submit_gate",
            "pipeline_version": "phase1_submission_analysis",
            "zip_path": str(archive_path),
            "zip_sha256": _sha256(archive_path),
            "error_count": sum(error_summary.values()),
            "notes": "Gold labels are unavailable; error_summary combines external metrics when supplied with local heuristic proxies.",
        },
        "phase1": {
            "metrics": phase1_metrics,
            "validation_summary": _validation_summary(validation_issues),
        },
        "profile": {
            **profile,
            "lab_dose_like_count": heuristic_counts.get("spurious_entity", 0),
            "likely_missing_historical_count": heuristic_counts.get("likely_missing_historical", 0),
            "likely_missing_negation_count": heuristic_counts.get("likely_missing_negation", 0),
            "likely_possible_context_count": heuristic_counts.get("likely_possible", 0),
            "coverage_low": _coverage_low(texts, all_items),
        },
        "candidate_metrics": {
            "codable_entities": heuristic_counts["codable_entities"],
            "codable_entities_with_one_candidate": heuristic_counts["codable_with_one_candidate"],
            "codable_entities_with_no_candidates": heuristic_counts["codable_with_no_candidates"],
        },
        "validation_issues": validation_issues,
        "error_summary": error_summary,
        "errors": errors[:120],
    }
    return report


def write_phase1_submission_analysis(report: dict[str, Any], output_dir: str | Path) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    _write_json(path / "external_grader_report.json", report)
    _write_jsonl(path / "errors.jsonl", _dict_list(report.get("errors", [])))
    (path / "analysis.md").write_text(render_phase1_submission_analysis(report), encoding="utf-8")


def render_phase1_submission_analysis(report: dict[str, Any]) -> str:
    summary = _mapping(report.get("summary", {}))
    phase1 = _mapping(report.get("phase1", {}))
    metrics = _mapping(phase1.get("metrics", {}))
    validation = _mapping(phase1.get("validation_summary", {}))
    profile = _mapping(report.get("profile", {}))
    error_summary = _mapping(report.get("error_summary", {}))
    lines = [
        "# Phase 1 Submission Analysis",
        "",
        f"- Zip: `{summary.get('zip_path', '')}`",
        f"- SHA-256: `{summary.get('zip_sha256', '')}`",
        f"- Validation issue count: `{validation.get('issue_count', 0)}`",
    ]
    if metrics:
        lines.append(
            "- External/local metrics: "
            f"score `{_fmt(metrics.get('score'))}`, "
            f"WER `{_fmt(metrics.get('wer'))}`, "
            f"J_assertion `{_fmt(metrics.get('assertions_score'))}`, "
            f"J_candidates `{_fmt(metrics.get('candidates_score'))}`"
        )
    lines.extend(
        [
            "",
            "## Local Profile",
            "",
            f"- Output entities: {profile.get('total_entities', 0)} across {profile.get('documents', 0)} files.",
            f"- Empty files: {profile.get('empty_files', [])}",
            f"- Type counts: {profile.get('type_counts', {})}",
            f"- Assertions: {profile.get('assertion_counts', {})}",
            f"- Candidate shape: {profile.get('candidate_count_by_type_len', {})}",
            "",
            "## Error Summary",
            "",
        ]
    )
    if error_summary:
        for key, value in sorted(error_summary.items(), key=lambda item: (-int(item[1]), item[0])):
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- No structured or heuristic errors.")
    lines.extend(["", "## Recommended Next Steps", ""])
    lines.extend(
        [
            "1. Fix section/subheading detection before tuning context rules further.",
            "2. Suppress medication-dose values from Phase 1 lab-result output.",
            "3. Expand dictionary aliases for high-miss terms listed in `coverage_low`.",
            "4. Run a controlled candidate-width experiment, for example `--max-candidates 5`.",
        ]
    )
    lines.extend(["", "## Example Errors", ""])
    for row in _dict_list(report.get("errors", []))[:20]:
        window = str(row.get("text_window", "")).replace("\n", " ")[:180]
        lines.append(
            f"- `{row.get('error_type')}` doc `{row.get('document_id')}` span `{row.get('span')}`: "
            f"{row.get('notes', '')} Window: {window}"
        )
    return "\n".join(lines) + "\n"


def _load_texts(input_dir: Path) -> dict[str, str]:
    return {path.stem: path.read_text(encoding="utf-8") for path in input_dir.glob("*.txt")}


def _load_zip_rows(zip_path: Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not name.endswith(".json"):
                continue
            payload = json.loads(archive.read(name).decode("utf-8"))
            rows[Path(name).stem] = payload if isinstance(payload, list) else []
    return dict(sorted(rows.items(), key=lambda item: _numeric_key(item[0])))


def _profile_rows(rows_by_document: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    type_counts: Counter[str] = Counter()
    assertion_counts: Counter[str] = Counter()
    candidate_counts: Counter[tuple[str, int]] = Counter()
    file_counts: dict[str, int] = {}
    all_items: list[tuple[str, dict[str, Any]]] = []
    for document_id, rows in rows_by_document.items():
        file_counts[document_id] = len(rows)
        for row in rows:
            entity_type = str(row.get("type", ""))
            type_counts[entity_type] += 1
            assertions = _list_value(row.get("assertions"))
            if assertions:
                assertion_counts.update(str(assertion) for assertion in assertions)
            else:
                assertion_counts["NONE"] += 1
            candidates = _list_value(row.get("candidates"))
            candidate_counts[(entity_type, len(candidates))] += 1
            all_items.append((document_id, row))
    counts = list(file_counts.values()) or [0]
    profile = {
        "documents": len(rows_by_document),
        "total_entities": len(all_items),
        "empty_files": [document_id for document_id, count in sorted(file_counts.items(), key=lambda item: _numeric_key(item[0])) if count == 0],
        "entity_count_min": min(counts),
        "entity_count_max": max(counts),
        "entity_count_mean": round(sum(counts) / len(counts), 2),
        "entity_count_median": statistics.median(counts),
        "type_counts": dict(type_counts),
        "assertion_counts": dict(assertion_counts),
        "candidate_count_by_type_len": {f"{item[0][0]}:{item[0][1]}": item[1] for item in candidate_counts.items()},
        "top_file_counts": sorted(file_counts.items(), key=lambda item: item[1], reverse=True)[:15],
        "low_file_counts": sorted(file_counts.items(), key=lambda item: item[1])[:20],
    }
    return profile, all_items


def _heuristic_errors(
    texts: dict[str, str],
    all_items: list[tuple[str, dict[str, Any]]],
    profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    errors: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for document_id in profile.get("empty_files", []):
        counts["missing_entity"] += 1
        errors.append(
            _error_row(
                document_id=str(document_id),
                stage="entity_extraction",
                error_type="phase1_missing_entity",
                severity="high",
                span=None,
                text_window=texts.get(str(document_id), "")[:240],
                prediction=[],
                notes="Output file is empty; likely missing Phase 1 entities.",
            )
        )
    for document_id, row in all_items:
        entity_type = str(row.get("type", ""))
        candidates = _list_value(row.get("candidates"))
        if entity_type in _CODABLE_TYPES:
            counts["codable_entities"] += 1
            counts["codable_with_one_candidate"] += int(len(candidates) == 1)
            counts["codable_with_no_candidates"] += int(len(candidates) == 0)
        text = str(row.get("text", ""))
        span = row.get("position")
        window = _window(texts.get(document_id, ""), span)
        if entity_type == "KẾT_QUẢ_XÉT_NGHIỆM" and _DOSE_LIKE_RE.match(text):
            counts["spurious_entity"] += 1
            errors.append(
                _error_row(
                    document_id=document_id,
                    stage="entity_extraction",
                    error_type="phase1_spurious_entity",
                    severity="medium",
                    span=span,
                    text_window=window,
                    prediction=row,
                    notes="Dose-like value exported as lab result.",
                )
            )
        if entity_type in _ASSERTABLE_TYPES:
            assertions = set(_list_value(row.get("assertions")))
            left = _left_context(texts.get(document_id, ""), span).lower()
            right = _right_context(texts.get(document_id, ""), span).lower()
            if not assertions and _has_historical_marker(left):
                counts["assertion_confusion"] += 1
                counts["likely_missing_historical"] += 1
                errors.append(_assertion_error(document_id, row, window, "isHistorical?", "Historical/preadmission context with empty assertion."))
            if "isNegated" not in assertions and any(marker in left for marker in _NEGATION_MARKERS):
                counts["assertion_confusion"] += 1
                counts["likely_missing_negation"] += 1
                errors.append(_assertion_error(document_id, row, window, "isNegated?", "Negation cue in local left context but isNegated is absent."))
            if not assertions and any(marker in left or marker in right for marker in _POSSIBLE_MARKERS):
                counts["assertion_confusion"] += 1
                counts["likely_possible"] += 1
                errors.append(_assertion_error(document_id, row, window, "possible/not-present?", "Uncertainty cue exists; Phase 1 export has no possible label."))
    return errors, dict(counts)


def _has_historical_marker(left_context: str) -> bool:
    if any(marker in left_context for marker in _HISTORICAL_FALSE_POSITIVE_MARKERS):
        return False
    return any(marker in left_context for marker in _HISTORICAL_MARKERS)


def _error_summary(profile: dict[str, Any], counts: dict[str, int], metrics: dict[str, float]) -> dict[str, int]:
    total_entities = int(profile.get("total_entities", 0))
    codable = counts.get("codable_entities", 0)
    summary = {
        "phase1_missing_entity": counts.get("missing_entity", 0),
        "phase1_spurious_entity": counts.get("spurious_entity", 0),
        "phase1_assertion_confusion": counts.get("assertion_confusion", 0),
    }
    if "wer" in metrics:
        summary["phase1_text_boundary"] = round(total_entities * (metrics["wer"] / 100.0))
    if "candidates_score" in metrics:
        summary["phase1_candidate_confusion"] = round(codable * (1.0 - (metrics["candidates_score"] / 100.0)))
    return {key: value for key, value in summary.items() if value > 0}


def _coverage_low(texts: dict[str, str], all_items: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    predicted: dict[str, list[str]] = defaultdict(list)
    for document_id, row in all_items:
        predicted[document_id].append(str(row.get("text", "")).lower())
    rows: list[dict[str, Any]] = []
    for group, terms in _COVERAGE_TERMS.items():
        for term in terms:
            pattern = _term_pattern(term)
            raw_occurrences = sum(len(pattern.findall(text)) for text in texts.values())
            if raw_occurrences == 0:
                continue
            predicted_occurrences = sum(sum(1 for mention in mentions if mention == term.lower()) for mentions in predicted.values())
            missed = max(0, raw_occurrences - predicted_occurrences)
            if missed:
                rows.append(
                    {
                        "group": group,
                        "term": term,
                        "raw_occurrences": raw_occurrences,
                        "predicted_occurrences": predicted_occurrences,
                        "missed_occurrences": missed,
                    }
                )
    return sorted(rows, key=lambda row: (-int(row["missed_occurrences"]), str(row["group"]), str(row["term"])))[:50]


def _phase1_metrics(external_metrics: dict[str, float] | None) -> dict[str, float]:
    if not external_metrics:
        return {}
    metrics = {key: float(value) for key, value in external_metrics.items() if value is not None}
    if "wer" in metrics and "text_score" not in metrics:
        metrics["text_score"] = round(max(0.0, 100.0 - metrics["wer"]), 6)
    if "j_assertion" in metrics and "assertions_score" not in metrics:
        metrics["assertions_score"] = metrics["j_assertion"]
    if "j_candidates" in metrics and "candidates_score" not in metrics:
        metrics["candidates_score"] = metrics["j_candidates"]
    return metrics


def _assertion_error(document_id: str, row: dict[str, Any], window: str, gold: str, notes: str) -> dict[str, Any]:
    return _error_row(
        document_id=document_id,
        stage="context",
        error_type="phase1_assertion_confusion",
        severity="high",
        span=row.get("position"),
        text_window=window,
        gold=gold,
        prediction=row,
        candidate_list=row.get("candidates", []),
        notes=notes,
    )


def _error_row(
    *,
    document_id: str,
    stage: str,
    error_type: str,
    severity: str,
    span: Any,
    text_window: str,
    prediction: Any,
    notes: str,
    gold: Any = None,
    candidate_list: list[Any] | None = None,
) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "stage": stage,
        "error_type": error_type,
        "severity": severity,
        "span": span,
        "text_window": text_window.replace("\n", " "),
        "gold": gold,
        "prediction": prediction,
        "candidate_rank": None,
        "candidate_list": candidate_list or [],
        "validation_path": None,
        "notes": notes,
    }


def _window(text: str, span: Any, radius: int = 85) -> str:
    if not _valid_span(span):
        return text[: 2 * radius]
    start, end = span
    return text[max(0, start - radius) : min(len(text), end + radius)]


def _left_context(text: str, span: Any, radius: int = 100) -> str:
    if not _valid_span(span):
        return ""
    start, _ = span
    return text[max(0, start - radius) : start]


def _right_context(text: str, span: Any, radius: int = 70) -> str:
    if not _valid_span(span):
        return ""
    _, end = span
    return text[end : min(len(text), end + radius)]


def _valid_span(span: Any) -> bool:
    return isinstance(span, list) and len(span) == 2 and all(isinstance(value, int) for value in span)


def _term_pattern(term: str) -> re.Pattern[str]:
    if " " in term or len(term) > 3:
        return re.compile(re.escape(term), flags=re.IGNORECASE | re.UNICODE)
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", flags=re.IGNORECASE | re.UNICODE)


def _validation_summary(issues: list[dict[str, str]]) -> dict[str, Any]:
    by_kind: Counter[str] = Counter(str(issue.get("kind", "")) for issue in issues)
    return {"issue_count": len(issues), "by_kind": dict(sorted(by_kind.items()))}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _numeric_key(value: str) -> tuple[int, str]:
    return (int(value), value) if value.isdigit() else (10**9, value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _fmt(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.6f}"
    return "N/A"
