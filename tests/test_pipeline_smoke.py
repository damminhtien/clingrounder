from medical_kg_nlp.datasets.synthetic_adapter import SyntheticDatasetAdapter
from medical_kg_nlp.pipeline.runner import PipelineRunner
from medical_kg_nlp.schema.types import AssertionStatus, RelationType


def test_pipeline_smoke_sample_note() -> None:
    document = SyntheticDatasetAdapter().load_documents("data/samples/sample_notes.jsonl")[0]
    prediction = PipelineRunner().process_document(document)
    by_text = {entity.text: entity for entity in prediction.entities}
    assert by_text["đái tháo đường type 2"].assertion == AssertionStatus.HISTORICAL
    assert by_text["viêm phổi"].assertion == AssertionStatus.POSSIBLE
    assert by_text["hen phế quản"].assertion == AssertionStatus.NEGATED
    assert by_text["ung thư phổi"].assertion == AssertionStatus.FAMILY
    assert by_text["metformin"].code == "6809"
    assert any(relation.type == RelationType.TREATS for relation in prediction.relations)


def test_pipeline_smoke_source_backed_treatment_seed() -> None:
    prediction = PipelineRunner().process_text(
        "source-backed-treatment",
        "Tăng huyết áp đang điều trị lisinopril.",
    )
    by_text = {entity.text: entity for entity in prediction.entities}

    assert by_text["Tăng huyết áp"].code == "I10"
    assert by_text["lisinopril"].code == "29046"
    assert any(relation.type == RelationType.TREATS for relation in prediction.relations)


def test_pipeline_phase1_sections_drive_historical_context_and_skip_dose_result() -> None:
    prediction = PipelineRunner().process_text(
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
    assert "25mg" not in by_text


def test_pipeline_phase1_preadmission_status_section_is_historical() -> None:
    prediction = PipelineRunner().process_text(
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
