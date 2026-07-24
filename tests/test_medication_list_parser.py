from __future__ import annotations

from medical_kg_nlp.ner.medication_list_parser import MedicationListParser
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.types import EntityType


def test_medication_list_parser_supports_numbered_bulleted_and_inline_items() -> None:
    text = (
        "Danh sách thuốc:\n"
        "- amlodipine 5 mg po daily điều trị tăng huyết áp\n"
        "• aspirin 81 mg po daily\n"
        "1) metformin 500 mg po bid\n"
        "Inline: 2. senna 8.6 mg po bid 3. clonazepam 1 mg po qhs"
    )

    items = MedicationListParser().items(text)
    medication_texts = [text[start:end] for start, end in (item.medication_span for item in items)]

    assert medication_texts == [
        "amlodipine 5 mg po daily",
        "aspirin 81 mg po daily",
        "metformin 500 mg po bid",
        "senna 8.6 mg po bid",
        "clonazepam 1 mg po qhs",
    ]
    assert text[slice(*items[0].indication_span)] == "tăng huyết áp"


def test_medication_list_adjudication_preserves_unseen_indication_entities() -> None:
    text = "1. amlodipine 5 mg po daily điều trị đau ngực và tăng huyết áp"
    drug = _entity(text, "amlodipine", EntityType.DRUG)
    symptom = _entity(text, "đau ngực", EntityType.SYMPTOM)
    diagnosis = _entity(text, "tăng huyết áp", EntityType.DISEASE)

    entities = MedicationListParser().adjudicate(text, [drug, symptom, diagnosis])

    assert entities == [drug, symptom, diagnosis]
    assert drug.medication_mention is not None
    full_span = drug.medication_mention.full_span
    assert text[full_span[0] : full_span[1]] == "amlodipine 5 mg po daily"
    drug.medication_mention.validate_offsets(text, drug.span)


def test_medication_list_does_not_materialize_hidden_indication_vocabulary() -> None:
    text = "1. guaifenesin po q6h:prn điều trị ho"
    drug = _entity(text, "guaifenesin", EntityType.DRUG)

    entities = MedicationListParser().adjudicate(text, [drug])

    assert entities == [drug]


def _entity(text: str, mention: str, entity_type: EntityType) -> EntityAnnotation:
    start = text.index(mention)
    return EntityAnnotation(
        id=mention,
        span=(start, start + len(mention)),
        text=mention,
        normalized_text=mention.casefold(),
        type=entity_type,
    )
