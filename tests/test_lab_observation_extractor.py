"""Focused contracts for anchored laboratory result extraction."""

from __future__ import annotations

from clingrounder.ner.lab_observation_extractor import LabObservationExtractor
from clingrounder.schema.annotation import EntityAnnotation
from clingrounder.schema.types import EntityType


def test_lab_observation_extracts_adjacent_qualitative_result_before_test() -> None:
    text = "dương tính cấy máu; âm tính HIDA; bình thường CT"
    anchors = [
        _anchor(text, "cấy máu"),
        _anchor(text, "HIDA"),
        _anchor(text, "CT"),
    ]

    results = LabObservationExtractor().extract(text, anchors)

    assert [entity.text for entity in results] == [
        "dương tính",
        "âm tính",
        "bình thường",
    ]
    assert all(text[start:end] == entity.text for entity in results for start, end in [entity.span])


def test_lab_observation_does_not_cross_words_to_find_left_qualifier() -> None:
    text = "Bệnh nhân trở lại bình thường trước chụp CT"

    results = LabObservationExtractor().extract(text, [_anchor(text, "CT")])

    assert results == []


def _anchor(text: str, mention: str) -> EntityAnnotation:
    start = text.index(mention)
    return EntityAnnotation(
        id=mention,
        span=(start, start + len(mention)),
        text=mention,
        normalized_text=mention.casefold(),
        type=EntityType.LAB_TEST,
    )
