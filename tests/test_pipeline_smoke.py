import json
from pathlib import Path

from medical_kg_nlp.datasets.synthetic_adapter import SyntheticDatasetAdapter
from medical_kg_nlp.pipeline.factory import PipelineFactory, PipelineFactoryConfig
from medical_kg_nlp.schema.types import AssertionStatus, EntityType, RelationType


def test_pipeline_smoke_sample_note() -> None:
    document = SyntheticDatasetAdapter().load_documents("data/samples/sample_notes.jsonl")[0]
    prediction = PipelineFactory.from_config().process_document(document)
    by_text = {entity.text: entity for entity in prediction.entities}
    assert by_text["đái tháo đường type 2"].assertion == AssertionStatus.HISTORICAL
    assert by_text["viêm phổi"].assertion == AssertionStatus.POSSIBLE
    assert by_text["hen phế quản"].assertion == AssertionStatus.NEGATED
    assert by_text["ung thư phổi"].assertion == AssertionStatus.FAMILY
    assert by_text["metformin"].code == "6809"
    assert any(relation.type == RelationType.TREATS for relation in prediction.relations)


def test_pipeline_smoke_source_backed_treatment_seed() -> None:
    prediction = PipelineFactory.from_config().process_text(
        "source-backed-treatment",
        "Tăng huyết áp đang điều trị lisinopril.",
    )
    by_text = {entity.text: entity for entity in prediction.entities}

    assert by_text["Tăng huyết áp"].code == "I10"
    assert by_text["lisinopril"].code == "29046"
    assert any(relation.type == RelationType.TREATS for relation in prediction.relations)


def test_pipeline_can_add_reviewed_vietnamese_clinical_lexicon(tmp_path: Path) -> None:
    additional_dictionary = tmp_path / "reviewed-public-fixture.jsonl"
    rows = [
        _local_concept("SYMPTOM_DYSURIA", "SYMPTOM", "tiểu buốt"),
        _local_concept("SYMPTOM_DYSPHAGIA", "SYMPTOM", "khó nuốt"),
        _local_concept("PROC_CORONARY_ANGIOGRAPHY", "PROCEDURE", "chụp mạch vành"),
    ]
    additional_dictionary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    runner = PipelineFactory.from_config(
        PipelineFactoryConfig(
            additional_recognition_dictionary_path=str(additional_dictionary)
        )
    )
    prediction = runner.process_text(
        "vn-lexicon-smoke",
        "Bệnh nhân có tiểu buốt và khó nuốt; đã thực hiện chụp mạch vành.",
    )
    by_text = {entity.text: entity for entity in prediction.entities}
    assert by_text["tiểu buốt"].type == EntityType.SYMPTOM
    assert by_text["tiểu buốt"].code == "SYMPTOM_DYSURIA"
    assert by_text["khó nuốt"].code == "SYMPTOM_DYSPHAGIA"
    assert by_text["chụp mạch vành"].type == EntityType.PROCEDURE


def _local_concept(code: str, semantic_type: str, alias: str) -> dict[str, object]:
    return {
        "concept_id": f"LOCAL:{code}",
        "code": code,
        "code_system": "LOCAL",
        "canonical_name": alias,
        "semantic_type": semantic_type,
        "aliases": [alias],
        "source": "synthetic_test_fixture",
    }


def test_pipeline_phase1_sections_drive_historical_context_and_skip_dose_result() -> None:
    prediction = PipelineFactory.from_config().process_text(
        "phase1-section-context",
        "1. Tiền sử bệnh\n"
        "Thuốc trước khi nhập viện\n"
        "- metoprolol 25mg po bid\n"
        "2. Bệnh sử hiện tại\n"
        "Triệu chứng hiện tại\n"
        "- đánh trống ngực\n",
    )
    by_text = {entity.text: entity for entity in prediction.entities}

    assert by_text["metoprolol"].assertion == AssertionStatus.HISTORICAL
    assert by_text["đánh trống ngực"].assertion == AssertionStatus.PRESENT
    assert by_text["25mg"].type == EntityType.STRENGTH
    assert by_text["25mg"].assertion == AssertionStatus.PRESENT
    assert any(
        relation.type == RelationType.HAS_DOSE
        and relation.head == by_text["metoprolol"].id
        and relation.tail == by_text["25mg"].id
        for relation in prediction.relations
    )


def test_pipeline_phase1_preadmission_status_section_is_historical() -> None:
    prediction = PipelineFactory.from_config().process_text(
        "phase1-preadmission-status",
        "1. Tiền sử bệnh hiện tại\n"
        "Tình trạng ngay trước khi nhập viện: Tiếp tục cảm thấy đánh trống ngực.\n"
        "2. Đánh giá tại bệnh viện\n"
        "Không ghi nhận đau ngực, khó thở, buồn nôn hoặc chóng mặt.\n",
    )
    by_text = {entity.text: entity for entity in prediction.entities}

    assert by_text["đánh trống ngực"].assertion == AssertionStatus.HISTORICAL
    assert by_text["đau ngực"].assertion == AssertionStatus.NEGATED
    assert by_text["khó thở"].assertion == AssertionStatus.NEGATED
    assert by_text["buồn nôn"].assertion == AssertionStatus.NEGATED
    assert by_text["chóng mặt"].assertion == AssertionStatus.NEGATED


def test_pipeline_phase1_chronic_conditions_section_overrides_possible_cue() -> None:
    prediction = PipelineFactory.from_config().process_text(
        "phase1-chronic-possible-history",
        "1. Tiền sử bệnh nội khoa\n"
        "Các bệnh lý mãn tính: nghi ngờ xơ gan do rượu\n"
        "2. Tiền sử bệnh hiện tại\n"
        "Triệu chứng hiện tại: hội chứng não gan\n",
    )
    by_text = {entity.text: entity for entity in prediction.entities}

    assert by_text["xơ gan do rượu"].assertion == AssertionStatus.HISTORICAL
    assert by_text["hội chứng não gan"].assertion == AssertionStatus.PRESENT
