from medical_kg_nlp.preprocessing.offset_mapping import collapse_whitespace_preserve_offsets
from medical_kg_nlp.preprocessing.normalizer import NormalizationContract


def test_offset_mapping_preserves_original_span() -> None:
    text = "Bệnh nhân không sốt."
    mapped = collapse_whitespace_preserve_offsets(text)
    start = mapped.normalized.index("sốt")
    original_span = mapped.normalized_span_to_original((start, start + len("sốt")))
    assert text[original_span[0] : original_span[1]] == "sốt"


def test_offset_mapping_trims_collapsed_edge_whitespace() -> None:
    text = "  abc  "
    mapped = collapse_whitespace_preserve_offsets(text)

    assert mapped.normalized == "abc"
    assert mapped.normalized_to_original == (2, 3, 4)
    assert mapped.normalized_span_to_original((0, 3)) == (2, 5)


def test_normalization_contract_preserves_source_coordinate_system() -> None:
    contract = NormalizationContract()
    source = "  Bệnh\n  nhân sốt.  "

    mapped = contract.prepare(source)
    start = mapped.normalized.index("sốt")
    source_span = mapped.normalized_span_to_original((start, start + len("sốt")))

    assert contract.downstream_uses_source_text is True
    assert source[source_span[0] : source_span[1]] == "sốt"
    assert contract.normalize_lookup_key("  SỐT! ") == "sốt"


def test_normalization_contract_rejects_non_monotonic_offset_map() -> None:
    mapped = collapse_whitespace_preserve_offsets("abc")
    invalid = type(mapped)(
        original=mapped.original,
        normalized=mapped.normalized,
        normalized_to_original=(0, 2, 1),
    )

    try:
        NormalizationContract.validate(invalid)
    except ValueError as error:
        assert "strictly increasing" in str(error)
    else:
        raise AssertionError("Expected malformed offset map to be rejected")
