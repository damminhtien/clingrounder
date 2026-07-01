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
