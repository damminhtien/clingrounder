from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.ner.dictionary_matcher import DictionaryMatcher
from medical_kg_nlp.schema.types import CodeSystem, EntityType


def test_dictionary_matcher_uses_toneless_matching_and_original_offsets() -> None:
    hypertension = _entry("ICD10:I10", "Tăng huyết áp", EntityType.DISEASE, "I10")
    abdominal_pain = _entry(
        "LOCAL:SYMPTOM_ABDOMINAL_PAIN", "Đau bụng", EntityType.SYMPTOM, "SYMPTOM_ABDOMINAL_PAIN"
    )
    pain = _entry("LOCAL:SYMPTOM_PAIN", "Đau", EntityType.SYMPTOM, "SYMPTOM_PAIN")
    prostate_cancer = _entry(
        "ICD10:C61",
        "u ác của tuyến tiền liệt",
        EntityType.DISEASE,
        "C61",
    )
    matcher = DictionaryMatcher(
        [
            ("Tăng huyết áp", hypertension),
            ("Đau bụng", abdominal_pain),
            ("Đau", pain),
            ("u ác của tuyến tiền liệt", prostate_cancer),
        ]
    )
    text = "Benh nhan tang huyet ap. Có đau bụng. Chẩn đoán u ác của tuyến tiền liệtAnh ấy ổn."

    matches = matcher.resolve_longest(matcher.find_candidates(text))
    by_text = {match.text: match for match in matches}

    assert by_text["tang huyet ap"].entry.concept_id == "ICD10:I10"
    assert by_text["tang huyet ap"].match_kind == "toneless"
    assert by_text["đau bụng"].entry.concept_id == "LOCAL:SYMPTOM_ABDOMINAL_PAIN"
    assert "đau" not in by_text
    assert by_text["u ác của tuyến tiền liệt"].entry.concept_id == "ICD10:C61"
    for match in matches:
        assert text[match.span[0] : match.span[1]] == match.text


def test_dictionary_matcher_blocks_lowercase_in_word_boundary() -> None:
    prostate_cancer = _entry(
        "ICD10:C61",
        "u ác của tuyến tiền liệt",
        EntityType.DISEASE,
        "C61",
    )
    matcher = DictionaryMatcher([("u ác của tuyến tiền liệt", prostate_cancer)])

    matches = matcher.resolve_longest(
        matcher.find_candidates("Chẩn đoán u ác của tuyến tiền liệtanh ấy đang chờ ghép thận.")
    )

    assert matches == []


def test_dictionary_matcher_does_not_merge_mentions_across_comma() -> None:
    difficult_swallowing = _entry(
        "LOCAL:SYMPTOM_DYSPHAGIA", "nuốt khó", EntityType.SYMPTOM, "SYMPTOM_DYSPHAGIA"
    )
    matcher = DictionaryMatcher([("nuốt khó", difficult_swallowing)])

    matches = matcher.resolve_longest(matcher.find_candidates("Không khó nuốt, khó thở."))

    assert matches == []


def test_dictionary_matcher_preserves_trailing_alias_parenthesis() -> None:
    hypertension = _entry(
        "ICD10:I10",
        "tăng huyết áp vô căn (nguyên phát)",
        EntityType.DISEASE,
        "I10",
    )
    matcher = DictionaryMatcher([("tăng huyết áp vô căn (nguyên phát)", hypertension)])
    text = "Tiền sử tăng huyết áp vô căn (nguyên phát)."

    matches = matcher.resolve_longest(matcher.find_candidates(text))

    assert matches[0].text == "tăng huyết áp vô căn (nguyên phát)"
    assert text[matches[0].span[0] : matches[0].span[1]] == matches[0].text


def test_dictionary_matcher_uses_weighted_interval_selection_for_two_concepts() -> None:
    broad = _entry("LOCAL:BROAD", "đau ngực khó thở", EntityType.SYMPTOM, "BROAD")
    chest_pain = _entry("LOCAL:CHEST_PAIN", "đau ngực", EntityType.SYMPTOM, "CHEST_PAIN")
    dyspnea = _entry("LOCAL:DYSPNEA", "khó thở", EntityType.SYMPTOM, "DYSPNEA")
    matcher = DictionaryMatcher(
        [
            ("đau ngực khó thở", broad),
            ("đau ngực", chest_pain),
            ("khó thở", dyspnea),
        ]
    )

    matches = matcher.resolve_longest(matcher.find_candidates("đau ngực khó thở"))

    assert [match.entry.concept_id for match in matches] == [
        "LOCAL:CHEST_PAIN",
        "LOCAL:DYSPNEA",
    ]


def _entry(concept_id: str, name: str, semantic_type: EntityType, code: str) -> ConceptEntry:
    return ConceptEntry(
        concept_id=concept_id,
        code=code,
        code_system=CodeSystem.ICD10 if semantic_type == EntityType.DISEASE else CodeSystem.LOCAL,
        canonical_name=name,
        semantic_type=semantic_type,
    )
