"""Model drug spans receive structured full-SIG metadata without sample lookup memory."""

from __future__ import annotations

from dataclasses import dataclass

from clingrounder.adapters.medication import MedicationMentionEntityExtractorAdapter
from clingrounder.schema.annotation import EntityAnnotation
from clingrounder.schema.types import EntityType
from clingrounder.utils.text import normalize_for_match


@dataclass(frozen=True)
class _Extractor:
    entity: EntityAnnotation

    def extract(self, source_text: str) -> list[EntityAnnotation]:
        del source_text
        return [self.entity]


def test_model_drug_span_is_decorated_with_full_medication_list_item() -> None:
    text = "Danh sách thuốc. 1. amlodipine 10 mg po daily điều trị tăng huyết áp"
    start = text.index("amlodipine")
    end = start + len("amlodipine")
    drug = EntityAnnotation(
        id="M1",
        span=(start, end),
        text=text[start:end],
        normalized_text=normalize_for_match(text[start:end]),
        type=EntityType.DRUG,
        confidence=0.9,
    )

    entities = MedicationMentionEntityExtractorAdapter(_Extractor(drug)).extract(text)

    assert len(entities) == 1
    medication = entities[0].medication_mention
    assert medication is not None
    assert text[medication.full_span[0] : medication.full_span[1]] == (
        "amlodipine 10 mg po daily"
    )
    medication.validate_offsets(text, drug.span)
