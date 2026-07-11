from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.ontology.phase1_assertions import (
    Phase1AssertionOverlay,
    load_phase1_assertion_overlays,
)
from medical_kg_nlp.ontology.phase1 import (
    PHASE1_ALLOWED_ASSERTIONS,
    PHASE1_ALLOWED_TYPES,
    PHASE1_ASSERTABLE_TYPES,
    PHASE1_CODABLE_TYPES,
    PHASE1_REQUIRED_KEYS,
    PHASE1_TYPE_BY_ENTITY_TYPE,
    expected_code_system,
)
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.document import ClinicalDocument
from medical_kg_nlp.schema.output import ClinicalPrediction
from medical_kg_nlp.schema.types import AssertionStatus, CodeSystem


_PHASE1_ASSERTION_BY_STATUS = {
    AssertionStatus.NEGATED: "isNegated",
    AssertionStatus.FAMILY: "isFamily",
    AssertionStatus.HISTORICAL: "isHistorical",
}
_PHASE1_ASSERTION_OVERLAYS: tuple[Phase1AssertionOverlay, ...] = load_phase1_assertion_overlays()
_DRUG_EXTENSION_MAX_CHARS = 96
_DRUG_EXTENSION_STOP_RE = re.compile(
    r"\s*(?:cho|vì|do|để|không|nhưng|tuy nhiên|with|for|due\s+to|because)\b",
    re.IGNORECASE,
)
_DRUG_EXTENSION_PATTERNS: tuple[tuple[re.Pattern[str], bool], ...] = (
    (
        re.compile(
            r"\s*(?:,?\s*)?\d+(?:[.,]\d+)?\s*"
            r"(?:mg|g|gram|mcg|microgram|ml|iu|đơn vị|units?)"
            r"(?:\s*/\s*(?:ngày|day|lần|dose))?",
            re.IGNORECASE,
        ),
        False,
    ),
    (
        re.compile(
            r"\s*(?:po|p\.o\.|iv|i\.v\.|im|sc|sl|uống|đường uống|tiêm tĩnh mạch|"
            r"tĩnh mạch|hít|nebs?|xịt|dán|nhỏ)\b",
            re.IGNORECASE,
        ),
        False,
    ),
    (
        re.compile(
            r"\s*(?:bid|tid|qid|qhs|q\d+h|q\s*\d+\s*h|daily|once|prn|hằng ngày|"
            r"hàng ngày|mỗi ngày|lần/ngày|lần mỗi ngày)\b",
            re.IGNORECASE,
        ),
        False,
    ),
    (re.compile(r"\s*x\s*\d+\b", re.IGNORECASE), False),
    (re.compile(r"\s+\d+\s*(?:lần|liều|viên)\b", re.IGNORECASE), False),
    (
        re.compile(r"\s+trong\s+\d+(?:[.,]\d+)?\s*(?:ngày|day|days|tuần|weeks?)\b", re.IGNORECASE),
        True,
    ),
    (
        re.compile(
            r"\s*,?\s*(?:sau đó|then|rồi)(?:\s+giảm\s+xuống)?\s+\d+(?:[.,]\d+)?\s*"
            r"(?:mg|g|gram|mcg|microgram|ml|iu|đơn vị|units?)"
            r"(?:\s*/\s*(?:ngày|day|lần|dose))?",
            re.IGNORECASE,
        ),
        True,
    ),
    (re.compile(r"\s+tại nhà\b", re.IGNORECASE), True),
    (re.compile(r"\s*\([^)\n\r]{1,50}\)"), True),
)

Phase1ExportPolicy = Literal["empty", "pipeline"]


@dataclass(frozen=True)
class Phase1ValidationIssue:
    kind: str
    path: str
    message: str
    document_id: str | None = None

    def to_json(self) -> dict[str, str]:
        payload = {"kind": self.kind, "path": self.path, "message": self.message}
        if self.document_id is not None:
            payload["document_id"] = self.document_id
        return payload


@dataclass(frozen=True)
class Phase1Match:
    gold_index: int
    pred_index: int
    text_score: float
    assertions_score: float
    candidates_score: float


def prediction_to_phase1_entities(
    prediction: ClinicalPrediction,
    *,
    max_candidates: int = 1,
    source_text: str | None = None,
    assertion_policy: Phase1ExportPolicy = "pipeline",
    candidate_policy: Phase1ExportPolicy = "pipeline",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entity in prediction.entities:
        phase1_type = PHASE1_TYPE_BY_ENTITY_TYPE.get(entity.type)
        if phase1_type is None:
            continue
        rows.append(
            entity_to_phase1(
                entity,
                phase1_type=phase1_type,
                max_candidates=max_candidates,
                source_text=source_text,
                assertion_policy=assertion_policy,
                candidate_policy=candidate_policy,
            )
        )
    return rows


def entity_to_phase1(
    entity: EntityAnnotation,
    *,
    phase1_type: str,
    max_candidates: int = 1,
    source_text: str | None = None,
    assertion_policy: Phase1ExportPolicy = "pipeline",
    candidate_policy: Phase1ExportPolicy = "pipeline",
) -> dict[str, Any]:
    _validate_export_policy(assertion_policy, "assertion_policy")
    _validate_export_policy(candidate_policy, "candidate_policy")
    text, span = _phase1_text_and_span(entity, phase1_type, source_text)
    return {
        "text": text,
        "type": phase1_type,
        "assertions": (
            _phase1_assertions(entity, source_text)
            if assertion_policy == "pipeline" and phase1_type in PHASE1_ASSERTABLE_TYPES
            else []
        ),
        "candidates": (
            _phase1_candidates(entity, phase1_type, max_candidates=max_candidates)
            if candidate_policy == "pipeline"
            else []
        ),
        "position": [span[0], span[1]],
    }


def validate_phase1_entities(
    rows: Any,
    source_text: str,
    *,
    document_id: str | None = None,
    dictionary: DictionaryStore | None = None,
) -> list[Phase1ValidationIssue]:
    issues: list[Phase1ValidationIssue] = []
    allowed_icd10, allowed_rxnorm = _allowed_codes(dictionary)
    if not isinstance(rows, list):
        return [
            Phase1ValidationIssue(
                kind="phase1_schema",
                path="$",
                message="Phase 1 output must be a JSON list, not an object.",
                document_id=document_id,
            )
        ]

    for index, item in enumerate(rows):
        path = f"$[{index}]"
        if not isinstance(item, dict):
            issues.append(
                _issue("phase1_schema", path, "Each Phase 1 item must be an object.", document_id)
            )
            continue
        issues.extend(
            _validate_phase1_item(
                item, source_text, path, document_id, allowed_icd10, allowed_rxnorm
            )
        )
    return issues


def validate_phase1_submission_dir(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    dictionary: DictionaryStore | None = None,
) -> list[Phase1ValidationIssue]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    issues: list[Phase1ValidationIssue] = []
    input_files = _phase1_input_files(input_path)
    expected_json_names = {f"{txt_path.stem}.json" for txt_path in input_files}
    for json_path in output_path.glob("*.json"):
        if json_path.name not in expected_json_names:
            issues.append(
                _issue(
                    "phase1_extra_output_file",
                    str(json_path),
                    "Output directory contains a JSON file without a matching input TXT file.",
                    json_path.stem,
                )
            )
    for txt_path in input_files:
        document_id = txt_path.stem
        json_path = output_path / f"{document_id}.json"
        if not json_path.exists():
            issues.append(
                _issue(
                    "phase1_missing_output_file",
                    str(json_path),
                    f"Missing output file for {txt_path.name}.",
                    document_id,
                )
            )
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            issues.append(_issue("phase1_schema", str(json_path), str(error), document_id))
            continue
        issues.extend(
            validate_phase1_entities(
                payload,
                txt_path.read_text(encoding="utf-8"),
                document_id=document_id,
                dictionary=dictionary,
            )
        )
    return issues


def validate_phase1_submission_zip(
    zip_path: str | Path,
    *,
    input_dir: str | Path | None = None,
    dictionary: DictionaryStore | None = None,
    expected_count: int = 100,
) -> list[Phase1ValidationIssue]:
    path = Path(zip_path)
    issues: list[Phase1ValidationIssue] = []
    if not path.exists():
        return [_issue("phase1_missing_zip", str(path), "Submission zip does not exist.", None)]
    with zipfile.ZipFile(path, "r") as archive:
        names = sorted(name for name in archive.namelist() if not name.endswith("/"))
        issues.extend(_zip_structure_issues(path, names, expected_count))
        if input_dir is not None:
            issues.extend(_zip_payload_issues(archive, input_dir, dictionary=dictionary))
    return issues


def _zip_structure_issues(
    zip_path: Path,
    names: list[str],
    expected_count: int,
) -> list[Phase1ValidationIssue]:
    expected = [f"output/{index}.json" for index in range(1, expected_count + 1)]
    if set(names) == set(expected) and len(names) == len(expected):
        return []
    expected_message = (
        f"Expected output/1.json..output/{expected_count}.json, "
        f"got {names[:3]}..{names[-3:] if names else []}."
    )
    return [
        _issue(
            "phase1_submission_structure",
            str(zip_path),
            expected_message,
            None,
        )
    ]


def _zip_payload_issues(
    archive: zipfile.ZipFile,
    input_dir: str | Path,
    *,
    dictionary: DictionaryStore | None,
) -> list[Phase1ValidationIssue]:
    issues: list[Phase1ValidationIssue] = []
    names = set(archive.namelist())
    for txt_path in _phase1_input_files(Path(input_dir)):
        document_id = txt_path.stem
        archive_name = f"output/{document_id}.json"
        if archive_name not in names:
            issues.append(
                _issue(
                    "phase1_missing_output_file",
                    archive_name,
                    f"Missing output file for {txt_path.name}.",
                    document_id,
                )
            )
            continue
        try:
            payload = json.loads(archive.read(archive_name).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            issues.append(_issue("phase1_schema", archive_name, str(error), document_id))
            continue
        issues.extend(
            validate_phase1_entities(
                payload,
                txt_path.read_text(encoding="utf-8"),
                document_id=document_id,
                dictionary=dictionary,
            )
        )
    return issues


def load_phase1_text_documents(input_dir: str | Path) -> list[ClinicalDocument]:
    return [
        ClinicalDocument(
            document_id=txt_path.stem,
            text=txt_path.read_text(encoding="utf-8"),
            metadata={"source_path": str(txt_path)},
        )
        for txt_path in _phase1_input_files(Path(input_dir))
    ]


def build_phase1_report(
    documents: list[ClinicalDocument],
    gold: list[ClinicalPrediction],
    predictions: list[ClinicalPrediction],
    dictionary: DictionaryStore | None = None,
    *,
    prediction_max_candidates: int = 1,
    gold_max_candidates: int = 50,
) -> dict[str, Any]:
    documents_by_id = {document.document_id: document for document in documents}
    gold_rows = _phase1_by_document(gold, max_candidates=gold_max_candidates)
    pred_rows = _phase1_by_document(predictions, max_candidates=prediction_max_candidates)
    validation_issues: list[dict[str, str]] = []
    for prediction in predictions:
        document = documents_by_id.get(prediction.document_id)
        if document is None:
            continue
        for issue in validate_phase1_entities(
            pred_rows.get(prediction.document_id, []),
            document.text,
            document_id=prediction.document_id,
            dictionary=dictionary,
        ):
            validation_issues.append(issue.to_json())
    metrics, errors = score_phase1_documents(gold_rows, pred_rows)
    validation_summary = _validation_summary(validation_issues)
    return {
        "metrics": metrics,
        "validation_issues": validation_issues,
        "validation_summary": validation_summary,
        "errors": errors,
        "gold": gold_rows,
        "predictions": pred_rows,
    }


def score_phase1_documents(
    gold_by_doc: dict[str, list[dict[str, Any]]],
    pred_by_doc: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document_ids = sorted(set(gold_by_doc) | set(pred_by_doc))
    text_scores: list[float] = []
    assertion_scores: list[float] = []
    candidate_scores: list[float] = []
    errors: list[dict[str, Any]] = []
    matched_entities = 0
    denominator = 0

    for document_id in document_ids:
        gold_rows = gold_by_doc.get(document_id, [])
        pred_rows = pred_by_doc.get(document_id, [])
        matches = _match_phase1_rows(gold_rows, pred_rows)
        matched_gold = {match.gold_index for match in matches}
        matched_pred = {match.pred_index for match in matches}
        denominator += len(gold_rows) + (len(pred_rows) - len(matched_pred))
        matched_entities += len(matches)

        for match in matches:
            text_scores.append(match.text_score)
            assertion_scores.append(match.assertions_score)
            candidate_scores.append(match.candidates_score)
            gold_row = gold_rows[match.gold_index]
            pred_row = pred_rows[match.pred_index]
            errors.extend(_phase1_match_errors(document_id, gold_row, pred_row, match))

        for index, row in enumerate(gold_rows):
            if index in matched_gold:
                continue
            text_scores.append(0.0)
            assertion_scores.append(0.0)
            candidate_scores.append(0.0)
            errors.append(
                _phase1_error(document_id, "phase1_missing_entity", gold=row, prediction=None)
            )

        for index, row in enumerate(pred_rows):
            if index in matched_pred:
                continue
            text_scores.append(0.0)
            assertion_scores.append(0.0)
            candidate_scores.append(0.0)
            errors.append(
                _phase1_error(document_id, "phase1_spurious_entity", gold=None, prediction=row)
            )

    if denominator == 0:
        text_score = assertion_score = candidate_score = 1.0
    else:
        text_score = _mean(text_scores)
        assertion_score = _mean(assertion_scores)
        candidate_score = _mean(candidate_scores)
    weighted = 0.3 * text_score + 0.3 * assertion_score + 0.4 * candidate_score
    metrics = {
        "score": round(weighted * 100.0, 6),
        "text_score": round(text_score, 6),
        "assertions_score": round(assertion_score, 6),
        "candidates_score": round(candidate_score, 6),
        "matched_entities": matched_entities,
        "scored_entities": denominator,
        "gold_entities": sum(len(rows) for rows in gold_by_doc.values()),
        "predicted_entities": sum(len(rows) for rows in pred_by_doc.values()),
        "weights": {"text": 0.3, "assertions": 0.3, "candidates": 0.4},
    }
    return metrics, errors


def write_phase1_output_dir(
    predictions: list[ClinicalPrediction],
    output_dir: str | Path,
    *,
    max_candidates: int = 1,
    clean: bool = True,
    source_text_by_document: Mapping[str, str] | None = None,
    assertion_policy: Phase1ExportPolicy = "pipeline",
    candidate_policy: Phase1ExportPolicy = "pipeline",
) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    if clean:
        for json_path in path.glob("*.json"):
            json_path.unlink()
    for prediction in predictions:
        rows = prediction_to_phase1_entities(
            prediction,
            max_candidates=max_candidates,
            source_text=source_text_by_document.get(prediction.document_id)
            if source_text_by_document
            else None,
            assertion_policy=assertion_policy,
            candidate_policy=candidate_policy,
        )
        (path / f"{prediction.document_id}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def zip_phase1_output_dir(output_dir: str | Path, zip_path: str | Path) -> None:
    output_path = Path(output_dir)
    archive_path = Path(zip_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for json_path in sorted(
            output_path.glob("*.json"), key=lambda item: _numeric_stem(item.stem)
        ):
            archive.write(json_path, arcname=f"output/{json_path.name}")


def phase1_validation_error_rows(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for issue in issues:
        rows.append(
            {
                "document_id": issue.get("document_id", ""),
                "stage": "phase1_submission",
                "error_type": issue.get("kind", "phase1_schema"),
                "severity": "blocking",
                "span": None,
                "text_window": "",
                "gold": None,
                "prediction": None,
                "candidate_rank": None,
                "candidate_list": [],
                "validation_path": issue.get("path", ""),
                "notes": issue.get("message", ""),
            }
        )
    return rows


def _validate_phase1_item(
    item: dict[str, Any],
    source_text: str,
    path: str,
    document_id: str | None,
    allowed_icd10: set[str],
    allowed_rxnorm: set[str],
) -> list[Phase1ValidationIssue]:
    issues: list[Phase1ValidationIssue] = []
    required = set(PHASE1_REQUIRED_KEYS)
    extra = sorted(set(item) - required)
    missing = sorted(required - set(item))
    for key in missing:
        issues.append(
            _issue("phase1_schema", f"{path}.{key}", "Missing required field.", document_id)
        )
    for key in extra:
        issues.append(
            _issue("phase1_extra_field", f"{path}.{key}", "Unexpected Phase 1 field.", document_id)
        )
    if missing:
        return issues

    text = item.get("text")
    phase1_type = item.get("type")
    assertions = item.get("assertions")
    candidates = item.get("candidates")
    position = item.get("position")

    if not isinstance(text, str) or not text:
        issues.append(
            _issue("phase1_schema", f"{path}.text", "text must be a non-empty string.", document_id)
        )
    if phase1_type not in PHASE1_ALLOWED_TYPES:
        issues.append(
            _issue(
                "phase1_invalid_type", f"{path}.type", f"Invalid type {phase1_type!r}.", document_id
            )
        )
    if (
        not isinstance(position, list)
        or len(position) != 2
        or not all(isinstance(value, int) for value in position)
    ):
        issues.append(
            _issue(
                "phase1_offset", f"{path}.position", "position must be [start, end].", document_id
            )
        )
    else:
        start, end = position
        if start < 0 or end <= start or end > len(source_text):
            issues.append(
                _issue(
                    "phase1_offset",
                    f"{path}.position",
                    f"Invalid position {position}.",
                    document_id,
                )
            )
        elif isinstance(text, str) and source_text[start:end] != text:
            issues.append(
                _issue(
                    "phase1_offset",
                    f"{path}.position",
                    "source_text[start:end] must exactly equal text.",
                    document_id,
                )
            )

    if not isinstance(assertions, list):
        issues.append(
            _issue(
                "phase1_invalid_assertion",
                f"{path}.assertions",
                "assertions must be a list.",
                document_id,
            )
        )
    else:
        if phase1_type not in PHASE1_ASSERTABLE_TYPES and assertions:
            issues.append(
                _issue(
                    "phase1_unexpected_assertions",
                    f"{path}.assertions",
                    "Only TRIỆU_CHỨNG, CHẨN_ĐOÁN, and THUỐC may contain assertions.",
                    document_id,
                )
            )
        if len(assertions) > 3:
            issues.append(
                _issue(
                    "phase1_invalid_assertion",
                    f"{path}.assertions",
                    "assertions may contain at most 3 items.",
                    document_id,
                )
            )
        for assertion in assertions:
            if assertion not in PHASE1_ALLOWED_ASSERTIONS:
                issues.append(
                    _issue(
                        "phase1_invalid_assertion",
                        f"{path}.assertions",
                        f"Invalid assertion {assertion!r}.",
                        document_id,
                    )
                )

    if not isinstance(candidates, list):
        issues.append(
            _issue(
                "phase1_invalid_candidates",
                f"{path}.candidates",
                "candidates must be a list.",
                document_id,
            )
        )
    else:
        issues.extend(
            _candidate_validation_issues(
                candidates, phase1_type, path, document_id, allowed_icd10, allowed_rxnorm
            )
        )
    return issues


def _candidate_validation_issues(
    candidates: list[Any],
    phase1_type: Any,
    path: str,
    document_id: str | None,
    allowed_icd10: set[str],
    allowed_rxnorm: set[str],
) -> list[Phase1ValidationIssue]:
    issues: list[Phase1ValidationIssue] = []
    if phase1_type not in PHASE1_CODABLE_TYPES and candidates:
        issues.append(
            _issue(
                "phase1_unexpected_candidates",
                f"{path}.candidates",
                "Only CHẨN_ĐOÁN and THUỐC may contain candidates.",
                document_id,
            )
        )
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate:
            issues.append(
                _issue(
                    "phase1_invalid_candidates",
                    f"{path}.candidates",
                    "Candidate codes must be strings.",
                    document_id,
                )
            )
            continue
        if phase1_type == "CHẨN_ĐOÁN" and allowed_icd10 and candidate not in allowed_icd10:
            issues.append(
                _issue(
                    "phase1_unknown_candidate",
                    f"{path}.candidates",
                    f"Unknown ICD-10 code {candidate!r}.",
                    document_id,
                )
            )
        if phase1_type == "THUỐC" and allowed_rxnorm and candidate not in allowed_rxnorm:
            issues.append(
                _issue(
                    "phase1_unknown_candidate",
                    f"{path}.candidates",
                    f"Unknown RxNorm code {candidate!r}.",
                    document_id,
                )
            )
    return issues


def _phase1_by_document(
    predictions: list[ClinicalPrediction],
    *,
    max_candidates: int,
) -> dict[str, list[dict[str, Any]]]:
    return {
        prediction.document_id: prediction_to_phase1_entities(
            prediction,
            max_candidates=max_candidates,
        )
        for prediction in predictions
    }


def _phase1_assertions(entity: EntityAnnotation, source_text: str | None) -> list[str]:
    labels: set[str] = set()
    statuses = set(entity.assertion_features.statuses()) or {entity.assertion}
    for status in statuses:
        value = _PHASE1_ASSERTION_BY_STATUS.get(status)
        if value:
            labels.add(value)
    if source_text is not None:
        for overlay in _PHASE1_ASSERTION_OVERLAYS:
            if overlay.matches(source_text, entity.span, entity_type=entity.type.value):
                labels.add(overlay.assertion)
    return [label for label in PHASE1_ALLOWED_ASSERTIONS if label in labels]


def _phase1_text_and_span(
    entity: EntityAnnotation,
    phase1_type: str,
    source_text: str | None,
) -> tuple[str, tuple[int, int]]:
    if phase1_type == "THUỐC" and source_text is not None:
        return _phase1_drug_text_and_span(entity, source_text)
    return entity.text, entity.span


def _phase1_drug_text_and_span(
    entity: EntityAnnotation, source_text: str
) -> tuple[str, tuple[int, int]]:
    start, end = entity.span
    if start < 0 or end <= start or end > len(source_text):
        return entity.text, entity.span
    if source_text[start:end] != entity.text:
        return entity.text, entity.span

    expanded_end = _phase1_drug_expanded_end(source_text, end)
    if expanded_end <= end:
        return entity.text, entity.span
    return source_text[start:expanded_end], (start, expanded_end)


def _phase1_drug_expanded_end(source_text: str, end: int) -> int:
    limit = min(len(source_text), end + _DRUG_EXTENSION_MAX_CHARS)
    cursor = end
    expanded_end = end
    while cursor < limit:
        tail = source_text[cursor:limit]
        if "\n" in tail[:1] or "\r" in tail[:1] or ";" in tail[:1]:
            break
        if _DRUG_EXTENSION_STOP_RE.match(source_text, cursor):
            break

        matched_end = None
        has_extension = expanded_end > end
        for pattern, requires_prior_extension in _DRUG_EXTENSION_PATTERNS:
            if requires_prior_extension and not has_extension:
                continue
            match = pattern.match(source_text, cursor)
            if match is None or match.end() <= cursor:
                continue
            token = source_text[cursor : match.end()]
            if not token.strip(" ,"):
                continue
            matched_end = match.end()
            break
        if matched_end is None:
            break
        cursor = matched_end
        expanded_end = matched_end

    while expanded_end > end and source_text[expanded_end - 1] in " ,":
        expanded_end -= 1
    return expanded_end


def _phase1_candidates(
    entity: EntityAnnotation, phase1_type: str, *, max_candidates: int
) -> list[str]:
    expected_system = _expected_code_system(phase1_type)
    if expected_system is None:
        return []
    codes: list[str] = []
    if entity.code_system == expected_system and entity.code:
        codes.append(entity.code)
    for candidate in entity.candidates:
        if candidate.code_system != expected_system or candidate.code is None:
            continue
        if candidate.code not in codes:
            codes.append(candidate.code)
        if len(codes) >= max_candidates:
            break
    return codes[:max_candidates]


def _expected_code_system(phase1_type: str) -> CodeSystem | None:
    return expected_code_system(phase1_type)


def _validate_export_policy(value: str, field: str) -> None:
    if value not in {"empty", "pipeline"}:
        raise ValueError(f"{field} must be one of: empty, pipeline.")


def _allowed_codes(dictionary: DictionaryStore | None) -> tuple[set[str], set[str]]:
    if dictionary is None:
        return set(), set()
    icd10 = {
        entry.code
        for entry in dictionary.entries
        if entry.code_system == CodeSystem.ICD10 and entry.code
    }
    rxnorm = {
        entry.code
        for entry in dictionary.entries
        if entry.code_system == CodeSystem.RXNORM and entry.code
    }
    return icd10, rxnorm


def _match_phase1_rows(
    gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]
) -> list[Phase1Match]:
    if not gold_rows or not pred_rows:
        return []
    weights = [[0.0 for _ in pred_rows] for _ in gold_rows]
    candidates: dict[tuple[int, int], Phase1Match] = {}
    for gold_index, gold in enumerate(gold_rows):
        for pred_index, pred in enumerate(pred_rows):
            if gold.get("type") != pred.get("type"):
                continue
            text_score = _text_similarity(str(gold.get("text", "")), str(pred.get("text", "")))
            span_overlap = _span_overlap(gold.get("position"), pred.get("position"))
            if text_score <= 0.0 and span_overlap <= 0.0:
                continue
            assertions_score = _jaccard(
                _string_set(gold.get("assertions")), _string_set(pred.get("assertions"))
            )
            candidates_score = _jaccard(
                _string_set(gold.get("candidates")), _string_set(pred.get("candidates"))
            )
            score = (
                (2.0 if _positions_equal(gold.get("position"), pred.get("position")) else 0.0)
                + text_score
                + span_overlap
            )
            weights[gold_index][pred_index] = score
            candidates[(gold_index, pred_index)] = Phase1Match(
                gold_index=gold_index,
                pred_index=pred_index,
                text_score=text_score,
                assertions_score=assertions_score,
                candidates_score=candidates_score,
            )
    return [
        candidates[(gold_index, pred_index)]
        for gold_index, pred_index in _maximum_weight_assignment(weights)
        if weights[gold_index][pred_index] > 0.0 and (gold_index, pred_index) in candidates
    ]


def _maximum_weight_assignment(weights: list[list[float]]) -> list[tuple[int, int]]:
    """Hungarian assignment with zero-weight dummy columns for abstention."""

    row_count = len(weights)
    real_column_count = len(weights[0]) if weights else 0
    if row_count == 0 or real_column_count == 0:
        return []
    column_count = real_column_count + row_count
    max_weight = max((max(row, default=0.0) for row in weights), default=0.0)
    costs = [
        [max_weight - weight for weight in row] + [max_weight for _ in range(row_count)]
        for row in weights
    ]

    row_potential = [0.0] * (row_count + 1)
    column_potential = [0.0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    predecessor = [0] * (column_count + 1)
    for row in range(1, row_count + 1):
        matched_row[0] = row
        column = 0
        minimum = [float("inf")] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[column] = True
            current_row = matched_row[column]
            delta = float("inf")
            next_column = 0
            for candidate_column in range(1, column_count + 1):
                if used[candidate_column]:
                    continue
                reduced_cost = (
                    costs[current_row - 1][candidate_column - 1]
                    - row_potential[current_row]
                    - column_potential[candidate_column]
                )
                if reduced_cost < minimum[candidate_column]:
                    minimum[candidate_column] = reduced_cost
                    predecessor[candidate_column] = column
                if minimum[candidate_column] < delta:
                    delta = minimum[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(column_count + 1):
                if used[candidate_column]:
                    row_potential[matched_row[candidate_column]] += delta
                    column_potential[candidate_column] -= delta
                else:
                    minimum[candidate_column] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous_column = predecessor[column]
            matched_row[column] = matched_row[previous_column]
            column = previous_column
            if column == 0:
                break

    assignment = [
        (matched_row[column] - 1, column - 1)
        for column in range(1, real_column_count + 1)
        if matched_row[column] > 0
    ]
    return sorted(assignment)


def _phase1_match_errors(
    document_id: str,
    gold: dict[str, Any],
    prediction: dict[str, Any],
    match: Phase1Match,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if not _positions_equal(gold.get("position"), prediction.get("position")) or gold.get(
        "text"
    ) != prediction.get("text"):
        errors.append(
            _phase1_error(document_id, "phase1_text_boundary", gold=gold, prediction=prediction)
        )
    if match.assertions_score < 1.0:
        errors.append(
            _phase1_error(
                document_id, "phase1_assertion_confusion", gold=gold, prediction=prediction
            )
        )
    if match.candidates_score < 1.0:
        errors.append(
            _phase1_error(
                document_id, "phase1_candidate_confusion", gold=gold, prediction=prediction
            )
        )
    return errors


def _phase1_error(
    document_id: str,
    error_type: str,
    *,
    gold: dict[str, Any] | None,
    prediction: dict[str, Any] | None,
    notes: str = "",
) -> dict[str, Any]:
    row = gold if gold is not None else prediction or {}
    return {
        "document_id": document_id,
        "stage": "phase1_submission",
        "error_type": error_type,
        "severity": "error",
        "span": row.get("position"),
        "text_window": row.get("text", ""),
        "gold": gold,
        "prediction": prediction,
        "candidate_rank": None,
        "candidate_list": prediction.get("candidates", []) if prediction is not None else [],
        "validation_path": "",
        "notes": notes,
    }


def _validation_summary(issues: list[dict[str, str]]) -> dict[str, Any]:
    by_kind: dict[str, int] = {}
    for issue in issues:
        kind = issue["kind"]
        by_kind[kind] = by_kind.get(kind, 0) + 1
    return {"issue_count": len(issues), "by_kind": dict(sorted(by_kind.items()))}


def _text_similarity(gold: str, prediction: str) -> float:
    gold_tokens = gold.split()
    pred_tokens = prediction.split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    distance = _edit_distance(gold_tokens, pred_tokens)
    return max(0.0, 1.0 - (distance / len(gold_tokens)))


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            substitution_cost = 0 if left_token == right_token else 1
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item) for item in value if isinstance(item, str)}


def _positions_equal(left: Any, right: Any) -> bool:
    return isinstance(left, list) and isinstance(right, list) and left == right


def _span_overlap(left: Any, right: Any) -> float:
    if not _valid_position(left) or not _valid_position(right):
        return 0.0
    left_start, left_end = cast(list[int], left)
    right_start, right_end = cast(list[int], right)
    overlap = max(0, min(left_end, right_end) - max(left_start, right_start))
    union = max(left_end, right_end) - min(left_start, right_start)
    return overlap / union if union > 0 else 0.0


def _valid_position(value: Any) -> bool:
    return (
        isinstance(value, list) and len(value) == 2 and all(isinstance(item, int) for item in value)
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 1.0


def _issue(kind: str, path: str, message: str, document_id: str | None) -> Phase1ValidationIssue:
    return Phase1ValidationIssue(kind=kind, path=path, message=message, document_id=document_id)


def _phase1_input_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*.txt"), key=lambda item: _numeric_stem(item.stem))


def _numeric_stem(value: str) -> tuple[int, str]:
    return (int(value), value) if value.isdigit() else (10**9, value)
