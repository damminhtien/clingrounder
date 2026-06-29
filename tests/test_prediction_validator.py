from __future__ import annotations

import copy
from typing import Any

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.schema.validator import PredictionValidator
from medical_kg_nlp.utils.io import read_jsonl


def test_prediction_validator_accepts_sample_gold() -> None:
    payload = _sample_prediction_payload()
    source_text = _sample_source_text()
    validator = PredictionValidator(DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl"))

    prediction, issues = validator.validate_payload(payload, source_text=source_text)

    assert prediction is not None
    assert issues == []


def test_prediction_validator_reports_offset_mismatch() -> None:
    payload = _sample_prediction_payload()
    payload["entities"][0]["span"] = [0, 5]
    validator = PredictionValidator()

    _, issues = validator.validate_payload(payload, source_text=_sample_source_text())

    assert [issue.kind for issue in issues] == ["offset"]


def test_prediction_validator_reports_invalid_code_system() -> None:
    payload = _sample_prediction_payload()
    payload["entities"][7]["code_system"] = "ICD-10"
    payload["entities"][7]["code"] = "E11"
    validator = PredictionValidator(DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl"))

    _, issues = validator.validate_payload(payload, source_text=_sample_source_text())

    assert any(issue.kind == "invalid_code_system" for issue in issues)


def test_prediction_validator_reports_unknown_dictionary_code() -> None:
    payload = _sample_prediction_payload()
    payload["entities"][0]["code"] = "MISSING"
    validator = PredictionValidator(DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl"))

    _, issues = validator.validate_payload(payload, source_text=_sample_source_text())

    assert any(issue.kind == "unknown_dictionary_code" for issue in issues)


def test_prediction_validator_reports_invalid_relation_reference() -> None:
    payload = _sample_prediction_payload()
    payload["relations"][0]["tail"] = "NO_SUCH_ENTITY"
    validator = PredictionValidator()

    _, issues = validator.validate_payload(payload, source_text=_sample_source_text())

    assert any(issue.kind == "invalid_relation" for issue in issues)


def test_prediction_validator_reports_schema_error() -> None:
    payload = _sample_prediction_payload()
    payload["entities"][0]["type"] = "ILLNESS"
    validator = PredictionValidator()

    prediction, issues = validator.validate_payload(payload, source_text=_sample_source_text())

    assert prediction is None
    assert [issue.kind for issue in issues] == ["schema"]


def _sample_prediction_payload() -> dict[str, Any]:
    return copy.deepcopy(read_jsonl("data/samples/gold.jsonl")[0])


def _sample_source_text() -> str:
    return str(read_jsonl("data/samples/sample_notes.jsonl")[0]["text"])
