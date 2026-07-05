from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.evaluation.phase1 import (
    build_phase1_report,
    load_phase1_text_documents,
    prediction_to_phase1_entities,
    score_phase1_documents,
    validate_phase1_entities,
    validate_phase1_submission_dir,
    validate_phase1_submission_zip,
    write_phase1_output_dir,
    zip_phase1_output_dir,
)
from medical_kg_nlp.evaluation.phase1_submission_analysis import build_phase1_submission_analysis
from medical_kg_nlp.schema.annotation import CandidateConcept, EntityAnnotation
from medical_kg_nlp.schema.document import ClinicalDocument
from medical_kg_nlp.schema.output import ClinicalPrediction
from medical_kg_nlp.schema.types import AssertionStatus, CodeSystem, EntityType


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
                candidates=[_candidate(CodeSystem.ICD10, "E11"), _candidate(CodeSystem.RXNORM, "6809")],
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
    assert rows[0]["candidates"] == []
    assert rows[1]["assertions"] == ["isNegated"]
    assert rows[1]["candidates"] == ["E11", "J18.9"]
    assert rows[2]["assertions"] == ["isHistorical"]
    assert rows[2]["candidates"] == ["6809"]
    assert rows[3]["assertions"] == []
    assert all(set(row) == {"text", "type", "assertions", "candidates", "position"} for row in rows)


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
            "position": [text.index("metoprolol"), text.index("metoprolol 25mg po bid") + len("metoprolol 25mg po bid")],
        }
    ]
    assert validate_phase1_entities(rows, text) == []


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
    assert rows[0]["position"] == [text.index("doxycycline"), text.index("doxycycline") + len("doxycycline")]
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
        [_entity("E1", text, "hen phế quản", EntityType.DISEASE, assertion=AssertionStatus.NEGATED)],
        [],
        "test",
    )

    rows = prediction_to_phase1_entities(prediction, source_text=text)

    assert rows[0]["assertions"] == ["isNegated", "isHistorical"]
    assert validate_phase1_entities(rows, text) == []


def test_prediction_to_phase1_entities_exports_negated_family_history_multilabel() -> None:
    text = "Không có tiền sử gia đình ung thư phổi."
    prediction = ClinicalPrediction.from_text(
        "1",
        text,
        [_entity("E1", text, "ung thư phổi", EntityType.DISEASE, assertion=AssertionStatus.NEGATED)],
        [],
        "test",
    )

    rows = prediction_to_phase1_entities(prediction, source_text=text)

    assert rows[0]["assertions"] == ["isNegated", "isFamily", "isHistorical"]
    assert validate_phase1_entities(rows, text) == []


def test_prediction_to_phase1_entities_does_not_treat_current_history_header_as_historical() -> None:
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

    rows = prediction_to_phase1_entities(prediction, source_text=text)
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
                    code="E11",
                    candidates=[_candidate(CodeSystem.ICD10, "J18.9")],
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
                    code="E11",
                    candidates=[_candidate(CodeSystem.ICD10, "J18.9")],
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
    assert validate_phase1_submission_zip(
        zip_path,
        input_dir=input_dir,
        dictionary=dictionary,
        expected_count=2,
    ) == []


def test_phase1_zip_structure_validation_is_order_independent(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    for index in range(1, 13):
        (output_dir / f"{index}.json").write_text("[]\n", encoding="utf-8")

    zip_path = tmp_path / "output.zip"
    zip_phase1_output_dir(output_dir, zip_path)

    assert validate_phase1_submission_zip(zip_path, expected_count=12) == []


def test_phase1_submission_cli_validates_zip(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "1.txt").write_text("Bệnh nhân ho.", encoding="utf-8")
    pred_path = tmp_path / "predictions.jsonl"
    text = (input_dir / "1.txt").read_text(encoding="utf-8")
    prediction = ClinicalPrediction.from_text(
        "1",
        text,
        [_entity("E1", text, "ho", EntityType.SYMPTOM)],
        [],
        "test",
    )
    pred_path.write_text(json.dumps(prediction.to_json(), ensure_ascii=False) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_phase1_submission.py",
            "--input-dir",
            str(input_dir),
            "--run-root",
            str(tmp_path / "runs"),
            "--output-dir",
            "phase1/output",
            "--pred",
            str(pred_path),
            "--zip",
            "phase1/output.zip",
            "--expected-count",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)
    output_dir = Path(summary["output_dir"])
    zip_path = Path(summary["zip"])
    assert Path(summary["run_dir"]).parent == tmp_path / "runs"

    subprocess.run(
        [
            sys.executable,
            "scripts/validate_phase1_submission.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--zip",
            str(zip_path),
            "--expected-count",
            "1",
        ],
        check=True,
    )


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


def test_phase1_submission_analysis_uses_word_boundaries_for_possible_markers(tmp_path: Path) -> None:
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
    return EntityAnnotation(
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


def _candidate(code_system: CodeSystem, code: str) -> CandidateConcept:
    return CandidateConcept(
        code_system=code_system,
        code=code,
        name=code,
        score=1.0,
        concept_id=f"{code_system.value}:{code}",
        source="test",
        matched_alias=code,
    )
