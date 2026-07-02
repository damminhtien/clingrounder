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
