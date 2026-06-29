# Offset Safety

Use this skill for preprocessing, normalization, tokenization, and span post-processing.

## Scope

- Check that all emitted entity spans index into original source text.
- Add tests for Unicode, Vietnamese diacritics, whitespace, and punctuation.
- Maintain offset maps for transformed text.
- Review NER and relation evidence spans for source-text compatibility.

## Guardrails

- Normalization must never replace source text for final offsets.
- Do not accept tests that only check normalized text.
- Run `tests/test_offset_mapping.py` and schema validation when spans change.
