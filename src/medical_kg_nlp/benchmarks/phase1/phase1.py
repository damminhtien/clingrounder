from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.ontology.phase1_assertions import (
    Phase1AssertionOverlay,
    load_phase1_assertion_overlays,
)
from medical_kg_nlp.ontology.phase1 import (
    PHASE1_ALLOWED_ASSERTIONS,
    PHASE1_ALLOWED_KEYS,
    PHASE1_ALLOWED_TYPES,
    PHASE1_ASSERTABLE_TYPES,
    PHASE1_CODABLE_TYPES,
    PHASE1_REQUIRED_KEYS,
    PHASE1_TYPE_BY_ENTITY_TYPE,
    expected_code_system,
)
from medical_kg_nlp.schema.annotation import AssertionEvidence, EntityAnnotation
from medical_kg_nlp.schema.document import ClinicalDocument
from medical_kg_nlp.schema.output import ClinicalPrediction
from medical_kg_nlp.schema.types import AssertionStatus, CodeSystem
from medical_kg_nlp.utils.text import normalize_for_match
from medical_kg_nlp.utils.io import read_source_text


_PHASE1_ASSERTION_BY_STATUS = {
    AssertionStatus.NEGATED: "isNegated",
    AssertionStatus.FAMILY: "isFamily",
    AssertionStatus.HISTORICAL: "isHistorical",
}
_PHASE1_ASSERTION_OVERLAYS: tuple[Phase1AssertionOverlay, ...] = load_phase1_assertion_overlays()
_SELECTIVE_EVIDENCE_SCOPES = frozenset({"left", "right", "bidirectional", "section_prior"})
Phase1ExportPolicy = Literal["empty", "pipeline", "selective"]


@dataclass(frozen=True)
class Phase1SelectiveExportConfig:
    assertion_allowed_scopes: frozenset[str]
    assertion_allowed_types: Mapping[str, frozenset[str]]
    assertion_min_evidence: int
    assertion_require_calibrated_evidence: bool
    calibrated_assertion_evidence: frozenset[tuple[str, str, str]]
    candidate_enabled: bool
    candidate_source_thresholds: Mapping[tuple[CodeSystem, str], float]
    candidate_require_reviewed: bool
    candidate_rxnorm_require_structured_mention: bool
    reviewed_candidates: frozenset[tuple[str, str, str]]
    candidate_selection_policy: Literal["unique", "expected_jaccard"] = "unique"
    candidate_max_candidates: int = 5
    candidate_min_expected_jaccard_gain: float = 0.0
    candidate_empty_probabilities: Mapping[CodeSystem, float] = field(default_factory=dict)
    candidate_rank_probabilities: Mapping[tuple[CodeSystem, str, int], float] = field(
        default_factory=dict
    )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        reviewed_candidates: frozenset[tuple[str, str, str]] = frozenset(),
        calibrated_assertion_evidence: frozenset[tuple[str, str, str]] = frozenset(),
    ) -> "Phase1SelectiveExportConfig":
        assertions = _required_mapping(payload, "assertions")
        candidates = _required_mapping(payload, "candidates")
        raw_types = _required_mapping(assertions, "allowed_types")
        unknown_labels = set(raw_types) - set(PHASE1_ALLOWED_ASSERTIONS)
        if unknown_labels:
            raise ValueError(
                f"selective.assertions.allowed_types has unknown labels: {sorted(unknown_labels)}"
            )
        allowed_types = {
            str(label): frozenset(_required_string_list(raw_types, str(label)))
            for label in PHASE1_ALLOWED_ASSERTIONS
        }
        invalid_types = sorted(
            {
                entity_type
                for entity_types in allowed_types.values()
                for entity_type in entity_types
                if entity_type not in PHASE1_ASSERTABLE_TYPES
            }
        )
        if invalid_types:
            raise ValueError(
                f"selective.assertions.allowed_types has invalid entity types: {invalid_types}"
            )
        source_thresholds = _candidate_source_thresholds(candidates)
        minimum = assertions.get("min_evidence", 1)
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
            raise ValueError("selective.assertions.min_evidence must be a positive integer")
        allowed_scopes = frozenset(_required_string_list(assertions, "allowed_scopes"))
        invalid_scopes = sorted(allowed_scopes - _SELECTIVE_EVIDENCE_SCOPES)
        if invalid_scopes:
            raise ValueError(
                f"selective.assertions.allowed_scopes has invalid scopes: {invalid_scopes}"
            )
        candidate_enabled = _required_bool(candidates, "enabled")
        require_calibrated_evidence = _required_bool(
            assertions, "require_calibrated_evidence"
        )
        if require_calibrated_evidence and not calibrated_assertion_evidence:
            raise ValueError(
                "selective assertions require a non-empty calibrated evidence map"
            )
        require_reviewed = _required_bool(candidates, "require_reviewed")
        selection_policy = str(candidates.get("selection_policy", "unique"))
        if selection_policy not in {"unique", "expected_jaccard"}:
            raise ValueError(
                "selective.candidates.selection_policy must be unique or expected_jaccard"
            )
        candidate_max_candidates = _positive_int(candidates, "max_candidates", default=5)
        minimum_gain = _probability(
            candidates.get("minimum_expected_jaccard_gain", 0.0),
            "selective.candidates.minimum_expected_jaccard_gain",
        )
        empty_probabilities = _candidate_empty_probabilities(candidates)
        rank_probabilities = _candidate_rank_probabilities(candidates)
        if selection_policy == "expected_jaccard" and (
            not empty_probabilities or not rank_probabilities
        ):
            raise ValueError(
                "expected_jaccard selection requires empty_probabilities and rank_probabilities"
            )
        if candidate_enabled and not source_thresholds:
            raise ValueError(
                "selective.candidates.source_thresholds must not be empty when candidates are enabled"
            )
        if candidate_enabled and require_reviewed and not reviewed_candidates:
            raise ValueError(
                "selective candidates require a non-empty reviewed candidate map"
            )
        return cls(
            assertion_allowed_scopes=allowed_scopes,
            assertion_allowed_types=allowed_types,
            assertion_min_evidence=minimum,
            assertion_require_calibrated_evidence=require_calibrated_evidence,
            calibrated_assertion_evidence=calibrated_assertion_evidence,
            candidate_enabled=candidate_enabled,
            candidate_source_thresholds=source_thresholds,
            candidate_require_reviewed=require_reviewed,
            candidate_rxnorm_require_structured_mention=_required_bool(
                candidates, "rxnorm_require_structured_mention"
            ),
            reviewed_candidates=reviewed_candidates,
            candidate_selection_policy=cast(
                Literal["unique", "expected_jaccard"], selection_policy
            ),
            candidate_max_candidates=candidate_max_candidates,
            candidate_min_expected_jaccard_gain=minimum_gain,
            candidate_empty_probabilities=empty_probabilities,
            candidate_rank_probabilities=rank_probabilities,
        )


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
    max_candidates: int = 5,
    source_text: str | None = None,
    assertion_policy: Phase1ExportPolicy = "pipeline",
    candidate_policy: Phase1ExportPolicy = "pipeline",
    selective_config: Phase1SelectiveExportConfig | None = None,
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
                selective_config=selective_config,
            )
        )
    return rows


def entity_to_phase1(
    entity: EntityAnnotation,
    *,
    phase1_type: str,
    max_candidates: int = 5,
    source_text: str | None = None,
    assertion_policy: Phase1ExportPolicy = "pipeline",
    candidate_policy: Phase1ExportPolicy = "pipeline",
    selective_config: Phase1SelectiveExportConfig | None = None,
) -> dict[str, Any]:
    _validate_export_policy(assertion_policy, "assertion_policy")
    _validate_export_policy(candidate_policy, "candidate_policy")
    if "selective" in {assertion_policy, candidate_policy} and selective_config is None:
        raise ValueError("selective export policy requires selective_config")
    text, span = _phase1_text_and_span(entity, phase1_type, source_text)
    row = {
        "text": text,
        "type": phase1_type,
        "assertions": (
            _phase1_assertions(entity, source_text)
            if assertion_policy == "pipeline" and phase1_type in PHASE1_ASSERTABLE_TYPES
            else _selective_assertions(entity, phase1_type, selective_config)
            if assertion_policy == "selective"
            else []
        ),
        "position": [span[0], span[1]],
    }
    if phase1_type in PHASE1_CODABLE_TYPES:
        # INVARIANT: the official executable specification requires this field for diagnosis and
        # medication, while omitting it for the three non-codable entity types.
        row["candidates"] = (
            _phase1_candidates(entity, phase1_type, max_candidates=max_candidates)
            if candidate_policy == "pipeline"
            else _selective_candidates(
                entity,
                phase1_type,
                selective_config,
                max_candidates=max_candidates,
            )
            if candidate_policy == "selective"
            else []
        )
    return row


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
                read_source_text(txt_path),
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
                read_source_text(txt_path),
                document_id=document_id,
                dictionary=dictionary,
            )
        )
    return issues


def load_phase1_text_documents(input_dir: str | Path) -> list[ClinicalDocument]:
    return [
        ClinicalDocument(
            document_id=txt_path.stem,
            text=read_source_text(txt_path),
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
    prediction_max_candidates: int = 5,
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
    max_candidates: int = 5,
    clean: bool = True,
    source_text_by_document: Mapping[str, str] | None = None,
    assertion_policy: Phase1ExportPolicy = "pipeline",
    candidate_policy: Phase1ExportPolicy = "pipeline",
    selective_config: Phase1SelectiveExportConfig | None = None,
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
            selective_config=selective_config,
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
            # ZipFile.write copies filesystem mtimes into the archive, which makes identical
            # submissions hash differently. Fixed metadata keeps probe manifests reproducible.
            info = zipfile.ZipInfo(
                filename=f"output/{json_path.name}",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, json_path.read_bytes())


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
    extra = sorted(set(item) - set(PHASE1_ALLOWED_KEYS))
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
    candidates = item.get("candidates", [])
    position = item.get("position")

    if phase1_type in PHASE1_CODABLE_TYPES and "candidates" not in item:
        issues.append(
            _issue(
                "phase1_schema",
                f"{path}.candidates",
                "candidates is required for CHẨN_ĐOÁN and THUỐC.",
                document_id,
            )
        )

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

    medication = entity.medication_mention
    if medication is None:
        return entity.text, entity.span
    try:
        medication.validate_offsets(source_text, entity.span)
    except ValueError:
        return entity.text, entity.span
    full_start, full_end = medication.full_span
    return source_text[full_start:full_end], medication.full_span


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
        if (
            not candidate.qualified
            or candidate.code_system != expected_system
            or candidate.code is None
        ):
            continue
        if candidate.code not in codes:
            codes.append(candidate.code)
        if len(codes) >= max_candidates:
            break
    return codes[:max_candidates]


def _selective_assertions(
    entity: EntityAnnotation,
    phase1_type: str,
    config: Phase1SelectiveExportConfig | None,
) -> list[str]:
    if config is None or phase1_type not in PHASE1_ASSERTABLE_TYPES:
        return []
    evidence_by_assertion: dict[str, list[AssertionEvidence]] = {}
    for item in entity.assertion_evidence:
        label = _PHASE1_ASSERTION_BY_STATUS.get(item.assertion)
        if label is None or item.scope not in config.assertion_allowed_scopes:
            continue
        if (
            config.assertion_require_calibrated_evidence
            and (item.rule_id, label, phase1_type)
            not in config.calibrated_assertion_evidence
        ):
            continue
        evidence_by_assertion.setdefault(label, []).append(item)
    return [
        label
        for label in PHASE1_ALLOWED_ASSERTIONS
        if phase1_type in config.assertion_allowed_types.get(label, frozenset())
        and len({item.rule_id for item in evidence_by_assertion.get(label, [])})
        >= config.assertion_min_evidence
    ]


def _selective_candidates(
    entity: EntityAnnotation,
    phase1_type: str,
    config: Phase1SelectiveExportConfig | None,
    *,
    max_candidates: int,
) -> list[str]:
    if config is None or not config.candidate_enabled:
        return []
    expected_system = _expected_code_system(phase1_type)
    if expected_system is None:
        return []
    if (
        expected_system == CodeSystem.RXNORM
        and config.candidate_rxnorm_require_structured_mention
        and not _has_structured_medication_mention(entity)
    ):
        return []
    eligible: list[tuple[str, str]] = []
    normalized = normalize_for_match(entity.text)
    for candidate in entity.candidates:
        threshold = config.candidate_source_thresholds.get(
            (expected_system, candidate.source)
        )
        if (
            threshold is None
            or not candidate.qualified
            or candidate.code_system != expected_system
            or candidate.code is None
            or candidate.emit_probability < threshold
        ):
            continue
        reviewed_key = (normalized, phase1_type, candidate.code)
        if config.candidate_require_reviewed and reviewed_key not in config.reviewed_candidates:
            continue
        if not any(code == candidate.code for code, _ in eligible):
            eligible.append((candidate.code, candidate.source))
    codes = [code for code, _ in eligible]
    if config.candidate_selection_policy == "unique":
        return codes if len(codes) == 1 else []

    probabilities: list[float] = []
    calibrated_codes: list[str] = []
    for rank, (code, source) in enumerate(eligible, start=1):
        probability = config.candidate_rank_probabilities.get(
            (expected_system, source, rank)
        )
        if probability is None:
            # Missing rank calibration means abstain from this rank and all lower ranks. Keeping a
            # contiguous prefix makes the policy deterministic and prevents cherry-picked codes.
            break
        calibrated_codes.append(code)
        probabilities.append(probability)
    empty_probability = config.candidate_empty_probabilities.get(expected_system)
    if empty_probability is None or not calibrated_codes:
        return []
    selected = _expected_jaccard_prefix_size(
        probabilities,
        empty_probability=empty_probability,
        max_candidates=min(max_candidates, config.candidate_max_candidates),
        minimum_gain=config.candidate_min_expected_jaccard_gain,
    )
    return calibrated_codes[:selected]


def _expected_jaccard_prefix_size(
    probabilities: list[float],
    *,
    empty_probability: float,
    max_candidates: int,
    minimum_gain: float,
) -> int:
    """Choose a ranked prefix using expected set Jaccard under calibrated marginals.

    Candidate inclusion events are treated as independent for this decision model. The explicit
    empty probability is calibrated separately because hidden-gold null prevalence is not implied
    reliably by alternative candidate marginals.
    """
    if max_candidates < 1:
        return 0
    best_size = 0
    best_score = empty_probability
    for size in range(1, min(max_candidates, len(probabilities)) + 1):
        selected_distribution = _bernoulli_count_distribution(probabilities[:size])
        omitted_distribution = _bernoulli_count_distribution(probabilities[size:])
        expected = 0.0
        for true_positive, true_positive_probability in enumerate(selected_distribution):
            for false_negative, false_negative_probability in enumerate(omitted_distribution):
                expected += (
                    true_positive_probability
                    * false_negative_probability
                    * true_positive
                    / (size + false_negative)
                )
        if expected > best_score:
            best_size = size
            best_score = expected
    return best_size if best_score >= empty_probability + minimum_gain else 0


def _bernoulli_count_distribution(probabilities: list[float]) -> list[float]:
    distribution = [1.0]
    for probability in probabilities:
        updated = [0.0] * (len(distribution) + 1)
        for count, mass in enumerate(distribution):
            updated[count] += mass * (1.0 - probability)
            updated[count + 1] += mass * probability
        distribution = updated
    return distribution


def _has_structured_medication_mention(entity: EntityAnnotation) -> bool:
    medication = entity.medication_mention
    if medication is None:
        return False
    return any(
        component.kind in {"administered_dose", "strength", "dose_form", "dosage"}
        for component in medication.components
    )


def _expected_code_system(phase1_type: str) -> CodeSystem | None:
    return expected_code_system(phase1_type)


def _validate_export_policy(value: str, field: str) -> None:
    if value not in {"empty", "pipeline", "selective"}:
        raise ValueError(f"{field} must be one of: empty, pipeline, selective.")


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"selective.{key} must be a mapping")
    return value


def _required_string_list(payload: Mapping[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"selective.{key} must be a list of strings")
    result = [str(item).strip() for item in value]
    if any(not item for item in result):
        raise ValueError(f"selective.{key} must not contain empty strings")
    return result


def _required_probability(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"selective.min_emit_probability.{key} must be a number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"selective.min_emit_probability.{key} must be between 0 and 1")
    return result


def _probability(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{path} must be a number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{path} must be between 0 and 1")
    return result


def _positive_int(payload: Mapping[str, Any], key: str, *, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"selective.candidates.{key} must be a positive integer")
    return value


def _candidate_empty_probabilities(
    candidates: Mapping[str, Any],
) -> dict[CodeSystem, float]:
    raw = candidates.get("empty_probabilities", {})
    if not isinstance(raw, Mapping):
        raise ValueError("selective.candidates.empty_probabilities must be a mapping")
    output: dict[CodeSystem, float] = {}
    for raw_system, value in raw.items():
        try:
            code_system = CodeSystem(str(raw_system))
        except ValueError as error:
            raise ValueError(
                f"Unknown candidate empty-probability code system {raw_system!r}"
            ) from error
        if code_system not in {CodeSystem.ICD10, CodeSystem.RXNORM}:
            raise ValueError(f"Unsupported candidate code system {code_system.value}")
        output[code_system] = _probability(
            value,
            f"selective.candidates.empty_probabilities.{code_system.value}",
        )
    return output


def _candidate_rank_probabilities(
    candidates: Mapping[str, Any],
) -> dict[tuple[CodeSystem, str, int], float]:
    raw_systems = candidates.get("rank_probabilities", {})
    if not isinstance(raw_systems, Mapping):
        raise ValueError("selective.candidates.rank_probabilities must be a mapping")
    output: dict[tuple[CodeSystem, str, int], float] = {}
    for raw_system, raw_sources in raw_systems.items():
        try:
            code_system = CodeSystem(str(raw_system))
        except ValueError as error:
            raise ValueError(
                f"Unknown candidate rank-probability code system {raw_system!r}"
            ) from error
        if code_system not in {CodeSystem.ICD10, CodeSystem.RXNORM}:
            raise ValueError(f"Unsupported candidate code system {code_system.value}")
        if not isinstance(raw_sources, Mapping):
            raise ValueError(
                f"selective.candidates.rank_probabilities.{code_system.value} must be a mapping"
            )
        for raw_source, raw_probabilities in raw_sources.items():
            source = str(raw_source).strip()
            if not source or "+" in source:
                raise ValueError("candidate rank probabilities require primary source names")
            if not isinstance(raw_probabilities, list) or not raw_probabilities:
                raise ValueError(
                    f"candidate rank probabilities for {code_system.value}:{source} "
                    "must be a non-empty list"
                )
            for rank, probability in enumerate(raw_probabilities, start=1):
                output[(code_system, source, rank)] = _probability(
                    probability,
                    f"selective.candidates.rank_probabilities."
                    f"{code_system.value}.{source}[{rank}]",
                )
    return output


def _candidate_source_thresholds(
    candidates: Mapping[str, Any],
) -> dict[tuple[CodeSystem, str], float]:
    raw_systems = _required_mapping(candidates, "source_thresholds")
    expected_keys = {CodeSystem.ICD10.value, CodeSystem.RXNORM.value}
    unknown = set(raw_systems) - expected_keys
    if unknown:
        raise ValueError(
            f"selective.candidates.source_thresholds has unknown code systems: {sorted(unknown)}"
        )
    result: dict[tuple[CodeSystem, str], float] = {}
    for code_system in (CodeSystem.ICD10, CodeSystem.RXNORM):
        by_source = _required_mapping(raw_systems, code_system.value)
        for raw_source in sorted(by_source):
            source = str(raw_source).strip()
            if not source or "+" in source:
                raise ValueError(
                    "selective candidate source names must be non-empty primary sources"
                )
            result[(code_system, source)] = _required_probability(by_source, str(raw_source))
    return result


def _required_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"selective.{key} must be a boolean")
    return value


def load_reviewed_candidate_map(path: str | Path) -> frozenset[tuple[str, str, str]]:
    rows: set[tuple[str, str, str]] = set()
    code_by_mention: dict[tuple[str, str], str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_number}: expected an object")
            if payload.get("review_status") != "reviewed":
                continue
            mention = normalize_for_match(str(payload.get("normalized_mention", "")))
            entity_type = str(payload.get("entity_type", ""))
            candidate = str(payload.get("candidate", ""))
            if not mention or entity_type not in PHASE1_CODABLE_TYPES or not candidate:
                raise ValueError(f"{path}:{line_number}: invalid reviewed candidate row")
            expected_system = _expected_code_system(entity_type)
            if payload.get("code_system") != (
                expected_system.value if expected_system is not None else None
            ):
                raise ValueError(
                    f"{path}:{line_number}: code_system does not match entity_type"
                )
            mention_key = (mention, entity_type)
            previous = code_by_mention.get(mention_key)
            if previous is not None and previous != candidate:
                raise ValueError(
                    f"{path}:{line_number}: conflicting reviewed codes "
                    f"{previous!r} and {candidate!r} for {mention_key!r}"
                )
            code_by_mention[mention_key] = candidate
            rows.add((mention, entity_type, candidate))
    return frozenset(rows)


def load_calibrated_assertion_map(
    path: str | Path,
) -> frozenset[tuple[str, str, str]]:
    rows: set[tuple[str, str, str]] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_number}: expected an object")
            if payload.get("review_status") != "calibrated":
                continue
            rule_id = str(payload.get("rule_id", "")).strip()
            assertion = str(payload.get("assertion", ""))
            entity_type = str(payload.get("entity_type", ""))
            if (
                not rule_id
                or assertion not in PHASE1_ALLOWED_ASSERTIONS
                or entity_type not in PHASE1_ASSERTABLE_TYPES
            ):
                raise ValueError(f"{path}:{line_number}: invalid calibrated assertion row")
            rows.add((rule_id, assertion, entity_type))
    return frozenset(rows)


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
