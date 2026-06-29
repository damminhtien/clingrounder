from medical_kg_nlp.preprocessing.offset_mapping import collapse_whitespace_preserve_offsets


def test_offset_mapping_preserves_original_span() -> None:
    text = "Bệnh nhân không sốt."
    mapped = collapse_whitespace_preserve_offsets(text)
    start = mapped.normalized.index("sốt")
    original_span = mapped.normalized_span_to_original((start, start + len("sốt")))
    assert text[original_span[0] : original_span[1]] == "sốt"

