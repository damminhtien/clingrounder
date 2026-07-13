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


def test_prediction_validator_reports_unknown_candidate_dictionary_code() -> None:
    payload = _sample_prediction_payload()
    payload["entities"][0]["candidates"] = [
        {
            "concept_id": "ICD10:MISSING",
            "code_system": "ICD-10",
            "code": "MISSING",
            "name": "Missing disease",
            "retrieval_score": 0.9,
            "emit_probability": 0.9,
            "source": "test",
            "evidence_sources": ["test"],
            "matched_alias": "missing",
            "qualified": True,
            "qualification_reason": "test_candidate",
        }
    ]
    validator = PredictionValidator(DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl"))

    _, issues = validator.validate_payload(payload, source_text=_sample_source_text())

    assert any(
        issue.kind == "unknown_dictionary_code"
        and issue.path == "$.entities[0].candidates[0].code"
        for issue in issues
    )


def test_prediction_validator_reports_invalid_candidate_code_system() -> None:
    payload = _sample_prediction_payload()
    payload["entities"][7]["candidates"] = [
        {
            "concept_id": "ICD10:E11",
            "code_system": "ICD-10",
            "code": "E11",
            "name": "Type 2 diabetes mellitus",
            "retrieval_score": 0.9,
            "emit_probability": 0.9,
            "source": "test",
            "evidence_sources": ["test"],
            "matched_alias": "metformin",
            "qualified": True,
            "qualification_reason": "test_candidate",
        }
    ]
    validator = PredictionValidator(DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl"))

    _, issues = validator.validate_payload(payload, source_text=_sample_source_text())

    assert any(issue.kind == "invalid_candidate_code_system" for issue in issues)


def test_prediction_validator_requires_candidate_qualification_metadata() -> None:
    payload = _sample_prediction_payload()
    candidate = _valid_candidate_payload()
    del candidate["qualified"]
    payload["entities"][0]["candidates"] = [candidate]
    validator = PredictionValidator()

    prediction, issues = validator.validate_payload(payload, source_text=_sample_source_text())

    assert prediction is None
    assert [issue.kind for issue in issues] == ["schema"]


def test_prediction_validator_rejects_non_boolean_candidate_qualification() -> None:
    payload = _sample_prediction_payload()
    payload["entities"][0]["candidates"] = [_valid_candidate_payload()]
    payload["entities"][0]["candidates"][0]["qualified"] = "yes"
    validator = PredictionValidator()

    prediction, issues = validator.validate_payload(payload, source_text=_sample_source_text())

    assert prediction is None
    assert [issue.kind for issue in issues] == ["schema"]


def test_prediction_validator_rejects_legacy_candidate_score() -> None:
    payload = _sample_prediction_payload()
    candidate = _valid_candidate_payload()
    candidate["score"] = candidate.pop("retrieval_score")
    payload["entities"][0]["candidates"] = [candidate]

    prediction, issues = PredictionValidator().validate_payload(
        payload,
        source_text=_sample_source_text(),
    )

    assert prediction is None
    assert [issue.kind for issue in issues] == ["schema"]


def test_prediction_validator_requires_assertion_evidence_field() -> None:
    payload = _sample_prediction_payload()
    del payload["entities"][0]["assertion_evidence"]

    prediction, issues = PredictionValidator().validate_payload(
        payload,
        source_text=_sample_source_text(),
    )

    assert prediction is None
    assert [issue.kind for issue in issues] == ["schema"]


def test_prediction_validator_round_trips_assertion_evidence() -> None:
    payload = _sample_prediction_payload()
    payload["entities"][0]["assertion_evidence"] = [
        {
            "rule_id": "neg.no_evidence",
            "assertion": "NEGATED",
            "cue": "không có",
            "scope": "left",
        }
    ]

    prediction, issues = PredictionValidator().validate_payload(
        payload,
        source_text=_sample_source_text(),
    )

    assert issues == []
    assert prediction is not None
    assert prediction.entities[0].assertion_evidence[0].rule_id == "neg.no_evidence"


def test_prediction_validator_round_trips_structured_medication_mention() -> None:
    payload = _sample_prediction_payload()
    payload["entities"][7]["medication_mention"] = {
        "drug_span": [160, 169],
        "full_span": [160, 175],
        "components": [{"kind": "strength", "span": [170, 175]}],
    }
    validator = PredictionValidator()

    prediction, issues = validator.validate_payload(payload, source_text=_sample_source_text())

    assert issues == []
    assert prediction is not None
    medication = prediction.entities[7].medication_mention
    assert medication is not None
    assert medication.full_span == (160, 175)


def test_prediction_validator_rejects_unknown_medication_component_kind() -> None:
    payload = _sample_prediction_payload()
    payload["entities"][7]["medication_mention"] = {
        "drug_span": [160, 169],
        "full_span": [160, 175],
        "components": [{"kind": "unknown", "span": [170, 175]}],
    }
    validator = PredictionValidator()

    prediction, issues = validator.validate_payload(payload, source_text=_sample_source_text())

    assert prediction is None
    assert [issue.kind for issue in issues] == ["schema"]


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


def _valid_candidate_payload() -> dict[str, object]:
    return {
        "concept_id": "ICD10:E11",
        "code_system": "ICD-10",
        "code": "E11",
        "name": "Type 2 diabetes mellitus",
        "retrieval_score": 0.9,
        "emit_probability": 0.9,
        "source": "test",
        "evidence_sources": ["test"],
        "matched_alias": "đái tháo đường type 2",
        "qualified": True,
        "qualification_reason": "test_candidate",
    }
