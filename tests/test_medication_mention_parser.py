from __future__ import annotations

import pytest

from medical_kg_nlp.ner.medication_mention_parser import MedicationMentionParser


@pytest.mark.parametrize(
    ("text", "drug", "expected", "component_kinds"),
    [
        (
            "Dùng amoxicillin/clavulanate 875 mg tablet po bid.",
            "amoxicillin/clavulanate",
            "amoxicillin/clavulanate 875 mg tablet po bid",
            ("administered_dose", "dose_form", "route", "frequency"),
        ),
        (
            "Điều trị albuterolipratropium nebs x3 every 20 minutes.",
            "albuterolipratropium",
            "albuterolipratropium nebs x3 every 20 minutes",
            ("dose_form", "dosage", "frequency"),
        ),
        (
            "Uống sulfamethoxazole-trimethoprim 800/160 mg tablet.",
            "sulfamethoxazole-trimethoprim",
            "sulfamethoxazole-trimethoprim 800/160 mg tablet",
            ("administered_dose", "dose_form"),
        ),
        (
            "Tiêm methylprednisolone sodium succinate 125 mg IV.",
            "methylprednisolone sodium succinate",
            "methylprednisolone sodium succinate 125 mg IV",
            ("administered_dose", "route"),
        ),
    ],
)
def test_medication_parser_preserves_compound_strength_form_and_route(
    text: str,
    drug: str,
    expected: str,
    component_kinds: tuple[str, ...],
) -> None:
    start = text.index(drug)
    mention = MedicationMentionParser().parse(text, (start, start + len(drug)))

    assert text[mention.full_span[0] : mention.full_span[1]] == expected
    assert tuple(component.kind for component in mention.components) == component_kinds
    mention.validate_offsets(text, (start, start + len(drug)))


def test_medication_parser_stops_before_indication_clause() -> None:
    text = "Dùng doxycycline cho viêm phổi."
    start = text.index("doxycycline")

    mention = MedicationMentionParser().parse(
        text,
        (start, start + len("doxycycline")),
    )

    assert mention.full_span == mention.drug_span
    assert mention.components == ()


def test_medication_parser_does_not_treat_prior_duration_as_medication_component() -> None:
    text = "Bắt đầu dùng suboxone 3 tuần trước vì rẻ hơn."
    start = text.index("suboxone")

    mention = MedicationMentionParser().parse(text, (start, start + len("suboxone")))

    assert mention.full_span == mention.drug_span


@pytest.mark.parametrize(
    ("text", "drug", "expected"),
    [
        (
            "metoprolol succinate xl 50 mg po daily điều trị tăng huyết áp",
            "metoprolol succinate",
            "metoprolol succinate xl 50 mg po daily",
        ),
        (
            "nystatin oral suspension 5 ml po qid:prn điều trị đau nhức",
            "nystatin",
            "nystatin oral suspension 5 ml po qid:prn",
        ),
        (
            "acetaminophen 325-650 mg po q6h:prn điều trị sốt đau",
            "acetaminophen",
            "acetaminophen 325-650 mg po q6h:prn",
        ),
        (
            "guaifenesin ml po q6h:prn điều trị ho",
            "guaifenesin",
            "guaifenesin ml po q6h:prn",
        ),
    ],
)
def test_medication_parser_supports_btc_sig_grammar(
    text: str, drug: str, expected: str
) -> None:
    start = text.index(drug)
    mention = MedicationMentionParser().parse(text, (start, start + len(drug)))

    assert text[mention.full_span[0] : mention.full_span[1]] == expected
