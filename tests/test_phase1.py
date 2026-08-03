from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.benchmarks.phase1.phase1 import (
    Phase1SelectiveExportConfig,
    _maximum_weight_assignment,
    build_phase1_report,
    load_reviewed_candidate_map,
    load_phase1_text_documents,
    prediction_to_phase1_entities,
    score_phase1_documents,
    validate_phase1_entities,
    validate_phase1_submission_dir,
    validate_phase1_submission_zip,
    write_phase1_output_dir,
    zip_phase1_output_dir,
)
from medical_kg_nlp.benchmarks.phase1.assertion_overlays import Phase1AssertionOverlay
from medical_kg_nlp.benchmarks.phase1.phase1_submission_analysis import build_phase1_submission_analysis
from medical_kg_nlp.ner.medication_mention_parser import MedicationMentionParser
from medical_kg_nlp.schema.annotation import (
    AssertionEvidence,
    AssertionFeatures,
    CandidateConcept,
    EntityAnnotation,
)
from medical_kg_nlp.schema.document import ClinicalDocument
from medical_kg_nlp.schema.output import ClinicalPrediction
from medical_kg_nlp.schema.types import AssertionStatus, CodeSystem, EntityType
from medical_kg_nlp.utils.io import read_yaml


def test_prediction_to_phase1_entities_exports_flat_official_schema() -> None:
    text = "Bệnh nhân ho. Chẩn đoán đái tháo đường type 2. Đang dùng metformin."
    prediction = ClinicalPrediction.from_text(
        document_id="1",
        text=text,
        entities=[
            _entity("E1", text, "ho", EntityType.SYMPTOM),
            _entity(
                "E2",
                text,
                "đái tháo đường type 2",
                EntityType.DISEASE,
                assertion=AssertionStatus.NEGATED,
                code_system=CodeSystem.ICD10,
                code="E11",
                candidates=[
                    _candidate(CodeSystem.ICD10, "E11"),
                    _candidate(CodeSystem.ICD10, "J18.9"),
                ],
            ),
            _entity(
                "E3",
                text,
                "metformin",
                EntityType.DRUG,
                assertion=AssertionStatus.HISTORICAL,
                code_system=CodeSystem.RXNORM,
                code="6809",
                candidates=[
                    _candidate(CodeSystem.ICD10, "E11"),
                    _candidate(CodeSystem.RXNORM, "6809"),
                ],
            ),
            _entity(
                "E4",
                text,
                "Bệnh nhân",
                EntityType.LAB_TEST,
                assertion=AssertionStatus.HISTORICAL,
            ),
            _entity("E5", text, "Bệnh nhân", EntityType.PATIENT_INFO),
        ],
        relations=[],
        pipeline_version="test",
    )

    rows = prediction_to_phase1_entities(prediction, max_candidates=2)

    assert [row["type"] for row in rows] == ["TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC", "TÊN_XÉT_NGHIỆM"]
    assert "candidates" not in rows[0]
    assert rows[1]["assertions"] == ["isNegated"]
    assert rows[1]["candidates"] == ["E11", "J18.9"]
    assert rows[2]["assertions"] == ["isHistorical"]
    assert rows[2]["candidates"] == ["6809"]
    assert rows[3]["assertions"] == []
    assert "candidates" not in rows[3]
    assert all(
        set(row)
        == (
            {"text", "type", "assertions", "candidates", "position"}
            if row["type"] in {"THUỐC", "CHẨN_ĐOÁN"}
            else {"text", "type", "assertions", "position"}
        )
        for row in rows
    )


def test_prediction_to_phase1_entities_supports_entity_only_abstention() -> None:
    text = "Tiền sử tăng huyết áp."
    prediction = ClinicalPrediction.from_text(
        document_id="1",
        text=text,
        entities=[
            _entity(
                "E1",
                text,
                "tăng huyết áp",
                EntityType.DISEASE,
                assertion=AssertionStatus.HISTORICAL,
                code_system=CodeSystem.ICD10,
                code="I10",
                candidates=[_candidate(CodeSystem.ICD10, "I10")],
            )
        ],
        relations=[],
        pipeline_version="test",
    )

    rows = prediction_to_phase1_entities(
        prediction,
        assertion_policy="empty",
        candidate_policy="empty",
    )

    assert rows[0]["text"] == "tăng huyết áp"
    assert rows[0]["assertions"] == []
    assert rows[0]["candidates"] == []


def test_selective_export_uses_evidence_and_reviewed_candidate_probability() -> None:
    text = "Tiền sử không có tăng huyết áp."
    entity = _entity(
        "E1",
        text,
        "tăng huyết áp",
        EntityType.DISEASE,
        assertion=AssertionStatus.NEGATED,
        candidates=[
            _candidate(
                CodeSystem.ICD10,
                "I10",
                source="exact",
                emit_probability=0.97,
            )
        ],
    )
    entity.assertion_evidence = (
        AssertionEvidence("NEG_NO", AssertionStatus.NEGATED, "không có", "left"),
        AssertionEvidence(
            "HIST_SECTION",
            AssertionStatus.HISTORICAL,
            "tiền sử",
            "section_prior",
        ),
    )
    prediction = ClinicalPrediction.from_text("1", text, [entity], [], "test")
    config = _selective_config(
        reviewed=frozenset({("tăng huyết áp", "CHẨN_ĐOÁN", "I10")})
    )

    rows = prediction_to_phase1_entities(
        prediction,
        assertion_policy="selective",
        candidate_policy="selective",
        selective_config=config,
    )

    assert rows[0]["assertions"] == ["isNegated"]
    assert rows[0]["candidates"] == ["I10"]


def test_selective_candidate_export_abstains_for_low_probability_or_ambiguity() -> None:
    text = "Chẩn đoán tăng huyết áp."
    entity = _entity(
        "E1",
        text,
        "tăng huyết áp",
        EntityType.DISEASE,
        candidates=[
            _candidate(
                CodeSystem.ICD10,
                "I10",
                source="exact",
                emit_probability=0.94,
            ),
            _candidate(
                CodeSystem.ICD10,
                "I11",
                source="exact",
                emit_probability=0.99,
            ),
        ],
    )
    prediction = ClinicalPrediction.from_text("1", text, [entity], [], "test")
    reviewed = frozenset(
        {
            ("tăng huyết áp", "CHẨN_ĐOÁN", "I10"),
            ("tăng huyết áp", "CHẨN_ĐOÁN", "I11"),
        }
    )

    rows = prediction_to_phase1_entities(
        prediction,
        assertion_policy="selective",
        candidate_policy="selective",
        selective_config=_selective_config(reviewed=reviewed),
    )

    assert rows[0]["candidates"] == ["I11"]

    entity.candidates[0] = _candidate(
        CodeSystem.ICD10,
        "I10",
        source="exact",
        emit_probability=0.99,
    )
    rows = prediction_to_phase1_entities(
        prediction,
        assertion_policy="selective",
        candidate_policy="selective",
        selective_config=_selective_config(reviewed=reviewed),
    )
    assert rows[0]["candidates"] == []


def test_selective_candidate_expected_jaccard_can_emit_dynamic_top_k() -> None:
    text = "Chẩn đoán tăng huyết áp."
    entity = _entity(
        "E1",
        text,
        "tăng huyết áp",
        EntityType.DISEASE,
        candidates=[
            _candidate(CodeSystem.ICD10, "I10", source="exact", emit_probability=0.99),
            _candidate(CodeSystem.ICD10, "I11", source="exact", emit_probability=0.99),
            _candidate(CodeSystem.ICD10, "I12", source="exact", emit_probability=0.99),
        ],
    )
    reviewed = frozenset(
        ("tăng huyết áp", "CHẨN_ĐOÁN", code) for code in ("I10", "I11", "I12")
    )
    config = _selective_config(
        reviewed=reviewed,
        selection_policy="expected_jaccard",
        empty_probability=0.05,
        rank_probabilities=(0.9, 0.8, 0.1),
    )

    rows = prediction_to_phase1_entities(
        ClinicalPrediction.from_text("1", text, [entity], [], "test"),
        candidate_policy="selective",
        selective_config=config,
        max_candidates=5,
    )

    assert rows[0]["candidates"] == ["I10", "I11"]


def test_selective_candidate_expected_jaccard_abstains_for_high_null_prevalence() -> None:
    text = "Chẩn đoán tăng huyết áp."
    entity = _entity(
        "E1",
        text,
        "tăng huyết áp",
        EntityType.DISEASE,
        candidates=[
            _candidate(CodeSystem.ICD10, "I10", source="exact", emit_probability=0.99)
        ],
    )
    config = _selective_config(
        reviewed=frozenset({("tăng huyết áp", "CHẨN_ĐOÁN", "I10")}),
        selection_policy="expected_jaccard",
        empty_probability=0.8,
        rank_probabilities=(0.6,),
    )

    rows = prediction_to_phase1_entities(
        ClinicalPrediction.from_text("1", text, [entity], [], "test"),
        candidate_policy="selective",
        selective_config=config,
    )

    assert rows[0]["candidates"] == []


def test_selective_export_requires_explicit_config() -> None:
    text = "Ho."
    prediction = ClinicalPrediction.from_text(
        "1", text, [_entity("E1", text, "Ho", EntityType.SYMPTOM)], [], "test"
    )

    try:
        prediction_to_phase1_entities(prediction, assertion_policy="selective")
    except ValueError as error:
        assert str(error) == "selective export policy requires selective_config"
    else:
        raise AssertionError("selective export accepted missing configuration")


def test_load_reviewed_candidate_map_accepts_only_reviewed_rows(tmp_path: Path) -> None:
    path = tmp_path / "reviewed.jsonl"
    path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "normalized_mention": "Tăng huyết áp",
                        "entity_type": "CHẨN_ĐOÁN",
                        "candidate": "I10",
                        "code_system": "ICD-10",
                        "review_status": "reviewed",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "normalized_mention": "viêm phổi",
                        "entity_type": "CHẨN_ĐOÁN",
                        "candidate": "J18.9",
                        "code_system": "ICD-10",
                        "review_status": "draft",
                    },
                    ensure_ascii=False,
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_reviewed_candidate_map(path) == frozenset(
        {("tăng huyết áp", "CHẨN_ĐOÁN", "I10")}
    )


def test_load_reviewed_candidate_map_rejects_conflicting_codes(tmp_path: Path) -> None:
    path = tmp_path / "reviewed.jsonl"
    rows = [
        {
            "normalized_mention": "tăng huyết áp",
            "entity_type": "CHẨN_ĐOÁN",
            "candidate": code,
            "code_system": "ICD-10",
            "review_status": "reviewed",
        }
        for code in ("I10", "I11")
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    try:
        load_reviewed_candidate_map(path)
    except ValueError as error:
        assert "conflicting reviewed codes" in str(error)
    else:
        raise AssertionError("conflicting reviewed candidate map was accepted")


def test_prediction_to_phase1_entities_exports_only_qualified_candidates() -> None:
    text = "Chẩn đoán tăng huyết áp."
    prediction = ClinicalPrediction.from_text(
        document_id="1",
        text=text,
        entities=[
            _entity(
                "E1",
                text,
                "tăng huyết áp",
                EntityType.DISEASE,
                code_system=CodeSystem.ICD10,
                code="I99",
                candidates=[
                    _candidate(CodeSystem.ICD10, "I10", qualified=True),
                    _candidate(CodeSystem.ICD10, "I11", qualified=False),
                ],
            )
        ],
        relations=[],
        pipeline_version="test",
    )

    rows = prediction_to_phase1_entities(prediction, max_candidates=5)

    assert rows[0]["candidates"] == ["I10"]


def test_phase1_export_preserves_supported_multi_label_assertions() -> None:
    text = "Tiền sử không có viêm phổi."
    entity = _entity(
        "E1",
        text,
        "viêm phổi",
        EntityType.DISEASE,
        assertion=AssertionStatus.NEGATED,
    )
    entity.assertion_features = AssertionFeatures(negated=True, historical=True)
    prediction = ClinicalPrediction.from_text("1", text, [entity], [], "test")

    rows = prediction_to_phase1_entities(prediction, source_text=text)

    assert rows[0]["assertions"] == ["isNegated", "isHistorical"]


def test_prediction_to_phase1_entities_can_expand_drug_dose_span_from_source_text() -> None:
    text = "Bệnh nhân dùng metoprolol 25mg po bid, không cải thiện."
    prediction = ClinicalPrediction.from_text(
        "1",
        text,
        [
            _entity(
                "E1",
                text,
                "metoprolol",
                EntityType.DRUG,
                code_system=CodeSystem.RXNORM,
                code="6918",
                candidates=[_candidate(CodeSystem.RXNORM, "6918")],
            )
        ],
        [],
        "test",
    )

    rows = prediction_to_phase1_entities(prediction, max_candidates=1, source_text=text)

    assert rows == [
        {
            "text": "metoprolol 25mg po bid",
            "type": "THUỐC",
            "assertions": [],
            "candidates": ["6918"],
            "position": [
                text.index("metoprolol"),
                text.index("metoprolol 25mg po bid") + len("metoprolol 25mg po bid"),
            ],
        }
    ]
    assert validate_phase1_entities(rows, text) == []


def test_phase1_entity_matching_uses_maximum_weight_assignment() -> None:
    assignment = _maximum_weight_assignment(
        [
            [10.0, 9.0],
            [9.0, 0.0],
        ]
    )

    assert assignment == [(0, 1), (1, 0)]


def test_prediction_to_phase1_entities_does_not_expand_drug_into_reason_clause() -> None:
    text = "Bệnh nhân dùng doxycycline cho viêm phổi."
    prediction = ClinicalPrediction.from_text(
        "1",
        text,
        [_entity("E1", text, "doxycycline", EntityType.DRUG)],
        [],
        "test",
    )

    rows = prediction_to_phase1_entities(prediction, source_text=text)

    assert rows[0]["text"] == "doxycycline"
    assert rows[0]["position"] == [
        text.index("doxycycline"),
        text.index("doxycycline") + len("doxycycline"),
    ]
    assert validate_phase1_entities(rows, text) == []


def test_prediction_to_phase1_entities_expands_drug_duration_only_after_dose_context() -> None:
    text = "Đã dùng prednisone 40 mg/ngày trong 3 ngày, sau đó 30 mg/ngày."
    prediction = ClinicalPrediction.from_text(
        "1",
        text,
        [_entity("E1", text, "prednisone", EntityType.DRUG)],
        [],
        "test",
    )

    rows = prediction_to_phase1_entities(prediction, source_text=text)

    expected = "prednisone 40 mg/ngày trong 3 ngày, sau đó 30 mg/ngày"
    assert rows[0]["text"] == expected
    assert rows[0]["position"] == [text.index("prednisone"), text.index(expected) + len(expected)]
    assert validate_phase1_entities(rows, text) == []


def test_prediction_to_phase1_entities_does_not_expand_drug_into_prior_duration() -> None:
    text = "Bệnh nhân bắt đầu dùng suboxone 3 tuần trước vì rẻ hơn."
    prediction = ClinicalPrediction.from_text(
        "1",
        text,
        [_entity("E1", text, "suboxone", EntityType.DRUG)],
        [],
        "test",
    )

    rows = prediction_to_phase1_entities(prediction, source_text=text)

    assert rows[0]["text"] == "suboxone"
    assert rows[0]["position"] == [text.index("suboxone"), text.index("suboxone") + len("suboxone")]
    assert validate_phase1_entities(rows, text) == []


def test_prediction_to_phase1_entities_exports_negated_historical_multilabel() -> None:
    text = "Không có tiền sử hen phế quản."
    prediction = ClinicalPrediction.from_text(
        "1",
        text,
        [
            _entity(
                "E1", text, "hen phế quản", EntityType.DISEASE, assertion=AssertionStatus.NEGATED
            )
        ],
        [],
        "test",
    )

    rows = prediction_to_phase1_entities(
        prediction,
        source_text=text,
        assertion_overlays=(
            Phase1AssertionOverlay(
                assertion="isHistorical",
                entity_types=(EntityType.DISEASE.value,),
                left_regex=re.compile(r"tiền sử\s*$", re.IGNORECASE),
            ),
        ),
    )

    assert rows[0]["assertions"] == ["isNegated", "isHistorical"]
    assert validate_phase1_entities(rows, text) == []


def test_prediction_to_phase1_entities_exports_negated_family_history_multilabel() -> None:
    text = "Không có tiền sử gia đình ung thư phổi."
    prediction = ClinicalPrediction.from_text(
        "1",
        text,
        [
            _entity(
                "E1", text, "ung thư phổi", EntityType.DISEASE, assertion=AssertionStatus.NEGATED
            )
        ],
        [],
        "test",
    )

    rows = prediction_to_phase1_entities(
        prediction,
        source_text=text,
        assertion_overlays=(
            Phase1AssertionOverlay(
                assertion="isFamily",
                entity_types=(EntityType.DISEASE.value,),
                left_regex=re.compile(r"tiền sử gia đình\s*$", re.IGNORECASE),
            ),
            Phase1AssertionOverlay(
                assertion="isHistorical",
                entity_types=(EntityType.DISEASE.value,),
                left_regex=re.compile(r"tiền sử gia đình\s*$", re.IGNORECASE),
            ),
        ),
    )

    assert rows[0]["assertions"] == ["isNegated", "isFamily", "isHistorical"]
    assert validate_phase1_entities(rows, text) == []


def test_prediction_to_phase1_entities_does_not_treat_current_history_header_as_historical() -> (
    None
):
    text = "Tiền sử bệnh hiện tại Lý do nhập viện: đau bụng."
    prediction = ClinicalPrediction.from_text(
        "1",
        text,
        [_entity("E1", text, "đau bụng", EntityType.SYMPTOM, assertion=AssertionStatus.PRESENT)],
        [],
        "test",
    )

    rows = prediction_to_phase1_entities(prediction, source_text=text)

    assert rows[0]["assertions"] == []
    assert validate_phase1_entities(rows, text) == []


def test_prediction_to_phase1_entities_limits_allergy_history_overlay_to_drugs() -> None:
    text = "Dị ứng: Dị ứng furosemide. Tiền sử bệnh hiện tại Lý do nhập viện: khó thở."
    prediction = ClinicalPrediction.from_text(
        "1",
        text,
        [
            _entity("E1", text, "furosemide", EntityType.DRUG, assertion=AssertionStatus.PRESENT),
            _entity("E2", text, "khó thở", EntityType.SYMPTOM, assertion=AssertionStatus.PRESENT),
        ],
        [],
        "test",
    )

    rows = prediction_to_phase1_entities(
        prediction,
        source_text=text,
        assertion_overlays=(
            Phase1AssertionOverlay(
                assertion="isHistorical",
                entity_types=(EntityType.DRUG.value,),
                left_regex=re.compile(r"Dị ứng:\s*Dị ứng\s*$", re.IGNORECASE),
            ),
        ),
    )
    by_text = {row["text"]: row for row in rows}

    assert by_text["furosemide"]["assertions"] == ["isHistorical"]
    assert by_text["khó thở"]["assertions"] == []
    assert validate_phase1_entities(rows, text) == []


def test_validate_phase1_entities_blocks_schema_offset_and_candidate_violations() -> None:
    text = "Bệnh nhân ho."
    row = {
        "text": "ho",
        "type": "TRIỆU_CHỨNG",
        "assertions": ["isPossible"],
        "candidates": ["E11"],
        "position": [text.index("ho"), text.index("ho") + 2],
        "relations": [],
    }

    issues = validate_phase1_entities([row], text, document_id="1")
    kinds = {issue.kind for issue in issues}

    assert "phase1_extra_field" in kinds
    assert "phase1_invalid_assertion" in kinds
    assert "phase1_unexpected_candidates" in kinds

    lab_issues = validate_phase1_entities(
        [{**row, "type": "TÊN_XÉT_NGHIỆM", "assertions": ["isHistorical"], "candidates": []}],
        text,
        document_id="1",
    )
    assert "phase1_unexpected_assertions" in {issue.kind for issue in lab_issues}

    offset_issues = validate_phase1_entities(
        [{**row, "assertions": [], "candidates": [], "position": [0, 2]}],
        text,
        document_id="1",
    )
    assert {issue.kind for issue in offset_issues} >= {"phase1_extra_field", "phase1_offset"}


def test_score_phase1_documents_uses_official_weights_and_candidate_jaccard() -> None:
    gold = {
        "1": [
            {
                "text": "đái tháo đường",
                "type": "CHẨN_ĐOÁN",
                "assertions": ["isNegated"],
                "candidates": ["E11"],
                "position": [0, 13],
            }
        ]
    }
    exact_metrics, exact_errors = score_phase1_documents(gold, gold)
    assert exact_metrics["score"] == 100.0
    assert exact_errors == []

    pred = {
        "1": [
            {
                "text": "đái tháo",
                "type": "CHẨN_ĐOÁN",
                "assertions": [],
                "candidates": ["E11", "J18.9"],
                "position": [0, 8],
            }
        ]
    }
    metrics, errors = score_phase1_documents(gold, pred)

    assert 0.0 < metrics["score"] < 100.0
    assert metrics["candidates_score"] == 0.5
    assert {row["error_type"] for row in errors} == {
        "phase1_text_boundary",
        "phase1_assertion_confusion",
        "phase1_candidate_confusion",
    }


def test_build_phase1_report_separates_phase1_validation_from_internal_metrics() -> None:
    text = "Chẩn đoán đái tháo đường type 2."
    documents = [ClinicalDocument(document_id="1", text=text)]
    gold = [
        ClinicalPrediction.from_text(
            "1",
            text,
            [
                _entity(
                    "G1",
                    text,
                    "đái tháo đường type 2",
                    EntityType.DISEASE,
                    code_system=CodeSystem.ICD10,
                    code="I99",
                    candidates=[
                        _candidate(CodeSystem.ICD10, "E11"),
                        _candidate(CodeSystem.ICD10, "J18.9"),
                    ],
                )
            ],
            [],
            "gold",
        )
    ]
    prediction = [
        ClinicalPrediction.from_text(
            "1",
            text,
            [
                _entity(
                    "P1",
                    text,
                    "đái tháo đường type 2",
                    EntityType.DISEASE,
                    code_system=CodeSystem.ICD10,
                    code="I99",
                    candidates=[
                        _candidate(CodeSystem.ICD10, "E11"),
                        _candidate(CodeSystem.ICD10, "J18.9"),
                    ],
                )
            ],
            [],
            "pred",
        )
    ]

    report = build_phase1_report(documents, gold, prediction, prediction_max_candidates=1)

    assert report["metrics"]["score"] == 80.0
    assert report["predictions"]["1"][0]["candidates"] == ["E11"]
    assert report["gold"]["1"][0]["candidates"] == ["E11", "J18.9"]
    assert report["validation_summary"]["issue_count"] == 0


def test_phase1_output_dir_and_zip_validation(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "1.txt").write_text("Bệnh nhân ho.", encoding="utf-8")
    (input_dir / "2.txt").write_text("Dùng metformin.", encoding="utf-8")
    documents = load_phase1_text_documents(input_dir)
    predictions = [
        ClinicalPrediction.from_text(
            "1",
            documents[0].text,
            [_entity("E1", documents[0].text, "ho", EntityType.SYMPTOM)],
            [],
            "test",
        ),
        ClinicalPrediction.from_text(
            "2",
            documents[1].text,
            [
                _entity(
                    "E2",
                    documents[1].text,
                    "metformin",
                    EntityType.DRUG,
                    code_system=CodeSystem.RXNORM,
                    code="6809",
                )
            ],
            [],
            "test",
        ),
    ]
    dictionary = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")

    output_dir.mkdir()
    (output_dir / "999.json").write_text("[]\n", encoding="utf-8")
    write_phase1_output_dir(predictions, output_dir)
    zip_path = tmp_path / "output.zip"
    zip_phase1_output_dir(output_dir, zip_path)

    assert not (output_dir / "999.json").exists()
    assert validate_phase1_submission_dir(input_dir, output_dir, dictionary=dictionary) == []
    assert (
        validate_phase1_submission_zip(
            zip_path,
            input_dir=input_dir,
            dictionary=dictionary,
            expected_count=2,
        )
        == []
    )


def test_phase1_zip_structure_validation_is_order_independent(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    for index in range(1, 13):
        (output_dir / f"{index}.json").write_text("[]\n", encoding="utf-8")

    zip_path = tmp_path / "output.zip"
    zip_phase1_output_dir(output_dir, zip_path)

    assert validate_phase1_submission_zip(zip_path, expected_count=12) == []


def test_phase1_zip_is_deterministic_across_source_mtimes(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    source = output_dir / "1.json"
    source.write_text("[]\n", encoding="utf-8")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    zip_phase1_output_dir(output_dir, first)
    source.touch()
    zip_phase1_output_dir(output_dir, second)

    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(
        second.read_bytes()
    ).digest()


def test_phase1_configs_separate_entity_only_and_full_execution() -> None:
    entity_only = read_yaml("configs/phase1_submission.yaml")
    full = read_yaml("configs/phase1_full.yaml")
    experiment_root = Path("configs/benchmarks/phase1/experiments")
    selective = read_yaml(experiment_root / "legacy_selective.yaml")
    selective_candidates = read_yaml(
        experiment_root / "legacy_selective_candidates.yaml"
    )

    assert entity_only["mode"] == "entity_only"
    assert entity_only["assertion_policy"] == "empty"
    assert entity_only["candidate_policy"] == "empty"
    assert not any(
        entity_only["pipeline"][key] for key in entity_only["pipeline"] if key.startswith("enable_")
    )

    assert full["mode"] == "full"
    assert full["assertion_policy"] == "pipeline"
    assert full["candidate_policy"] == "pipeline"
    for key in (
        "enable_context",
        "enable_linking",
        "enable_candidate_reranking",
        "enable_entity_kg_validation",
    ):
        assert full["pipeline"][key] is True
    assert full["pipeline"]["enable_relations"] is False

    assert selective["mode"] == "selective"
    assert selective["assertion_policy"] == "selective"
    assert selective["candidate_policy"] == "selective"
    assert selective["selective"]["candidates"]["enabled"] is False
    assert selective["pipeline"]["enable_context"] is True

    candidate_config = selective_candidates["selective"]["candidates"]
    assertion_config = selective_candidates["selective"]["assertions"]
    assert candidate_config["enabled"] is True
    assert candidate_config["source_thresholds"] == {
        "ICD-10": {"dictionary_exact": 0.99505},
        "RxNorm": {"dictionary_exact": 0.989362},
    }
    assert str(candidate_config["reviewed_map"]).startswith("data/manual_gold/")
    assert str(assertion_config["calibrated_evidence_map"]).startswith(
        "data/manual_gold/"
    )
    assert selective_candidates["pipeline"]["link_emit_probabilities_by_source"] == {
        "ICD-10:dictionary_exact": 0.99505,
        "RxNorm:dictionary_exact": 0.989362,
    }
    assert assertion_config["require_calibrated_evidence"] is True


def test_stable_phase1_config_root_does_not_advertise_selective_mode() -> None:
    stable_configs = sorted(Path("configs").glob("phase1*.yaml"))

    assert stable_configs
    assert all(read_yaml(path).get("mode") != "selective" for path in stable_configs)


@pytest.mark.release
def test_phase1_pre_submit_gate_writes_analysis_and_loop_artifacts(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    gate_dir = tmp_path / "gate"
    input_dir.mkdir()
    text = "Bệnh nhân ho. Dùng metformin."
    (input_dir / "1.txt").write_text(text, encoding="utf-8")
    prediction = ClinicalPrediction.from_text(
        "1",
        text,
        [
            _entity("E1", text, "ho", EntityType.SYMPTOM),
            _entity(
                "E2",
                text,
                "metformin",
                EntityType.DRUG,
                code_system=CodeSystem.RXNORM,
                code="6809",
                candidates=[_candidate(CodeSystem.RXNORM, "6809")],
            ),
        ],
        [],
        "test",
    )
    write_phase1_output_dir([prediction], output_dir, max_candidates=3)
    zip_path = tmp_path / "output.zip"
    zip_phase1_output_dir(output_dir, zip_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase1_pre_submit_gate.py",
            "--input-dir",
            str(input_dir),
            "--zip",
            str(zip_path),
            "--output-dir",
            str(gate_dir),
            "--journal-dir",
            str(tmp_path / "journal"),
            "--expected-count",
            "1",
            "--experiment-id",
            "TEST_GATE",
            "--score",
            "14.5",
            "--wer",
            "81.0",
            "--j-assertion",
            "17.0",
            "--j-candidates",
            "9.0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)

    assert summary["valid"] is True
    assert (gate_dir / "analysis.md").exists()
    assert (gate_dir / "external_grader_report.json").exists()
    assert (gate_dir / "loop_report.json").exists()
    assert (tmp_path / "journal" / "experiment_notebook.md").exists()


def test_phase1_submission_analysis_uses_word_boundaries_for_possible_markers(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    text = (
        "Đặc điểm triệu chứng: đau ngực - Mức độ nghiêm trọng: N/A. "
        "Được điều trị vì nghi ngờ viêm phổi."
    )
    (input_dir / "1.txt").write_text(text, encoding="utf-8")
    rows = [
        {
            "text": "đau ngực",
            "type": "TRIỆU_CHỨNG",
            "assertions": [],
            "candidates": [],
            "position": [text.index("đau ngực"), text.index("đau ngực") + len("đau ngực")],
        },
        {
            "text": "viêm phổi",
            "type": "CHẨN_ĐOÁN",
            "assertions": [],
            "candidates": [],
            "position": [text.index("viêm phổi"), text.index("viêm phổi") + len("viêm phổi")],
        },
    ]
    (output_dir / "1.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    zip_path = tmp_path / "output.zip"
    zip_phase1_output_dir(output_dir, zip_path)

    report = build_phase1_submission_analysis(
        input_dir=input_dir,
        zip_path=zip_path,
        dictionary=DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl"),
        expected_count=1,
    )

    possible_errors = [
        error
        for error in report["errors"]
        if error["notes"] == "Uncertainty cue exists; Phase 1 export has no possible label."
    ]
    assert report["profile"]["likely_possible_context_count"] == 1
    assert [error["prediction"]["text"] for error in possible_errors] == ["viêm phổi"]


def _entity(
    entity_id: str,
    source_text: str,
    mention: str,
    entity_type: EntityType,
    *,
    assertion: AssertionStatus = AssertionStatus.PRESENT,
    code_system: CodeSystem = CodeSystem.NONE,
    code: str | None = None,
    candidates: list[CandidateConcept] | None = None,
) -> EntityAnnotation:
    start = source_text.index(mention)
    entity = EntityAnnotation(
        id=entity_id,
        span=(start, start + len(mention)),
        text=mention,
        normalized_text=mention.lower(),
        type=entity_type,
        assertion=assertion,
        code_system=code_system,
        code=code,
        confidence=1.0,
        candidates=candidates or [],
    )
    if entity_type == EntityType.DRUG:
        entity.medication_mention = MedicationMentionParser().parse(source_text, entity.span)
    return entity


def _candidate(
    code_system: CodeSystem,
    code: str,
    *,
    qualified: bool = True,
    source: str = "test",
    emit_probability: float = 1.0,
) -> CandidateConcept:
    return CandidateConcept(
        code_system=code_system,
        code=code,
        name=code,
        retrieval_score=1.0,
        emit_probability=emit_probability,
        concept_id=f"{code_system.value}:{code}",
        source=source,
        evidence_sources=(source,),
        matched_alias=code,
        qualified=qualified,
        qualification_reason="test_qualified" if qualified else "test_rejected",
    )


def _selective_config(
    *,
    reviewed: frozenset[tuple[str, str, str]],
    selection_policy: Literal["unique", "expected_jaccard"] = "unique",
    empty_probability: float = 0.0,
    rank_probabilities: tuple[float, ...] = (),
) -> Phase1SelectiveExportConfig:
    return Phase1SelectiveExportConfig(
        assertion_allowed_scopes=frozenset({"left", "right"}),
        assertion_allowed_types={
            "isNegated": frozenset({"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"}),
            "isFamily": frozenset(),
            "isHistorical": frozenset({"CHẨN_ĐOÁN", "THUỐC"}),
        },
        assertion_min_evidence=1,
        assertion_require_calibrated_evidence=False,
        calibrated_assertion_evidence=frozenset(),
        candidate_enabled=True,
        candidate_source_thresholds={
            (CodeSystem.ICD10, "exact"): 0.95,
            (CodeSystem.RXNORM, "exact"): 0.98,
        },
        candidate_require_reviewed=True,
        candidate_rxnorm_require_structured_mention=False,
        reviewed_candidates=reviewed,
        candidate_selection_policy=selection_policy,
        candidate_empty_probabilities=(
            {CodeSystem.ICD10: empty_probability} if rank_probabilities else {}
        ),
        candidate_rank_probabilities={
            (CodeSystem.ICD10, "exact", rank): probability
            for rank, probability in enumerate(rank_probabilities, start=1)
        },
    )
