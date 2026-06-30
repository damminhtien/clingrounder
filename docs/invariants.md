# Invariants

## Offset Safety

- Entity spans are always offsets into the original source text.
- Normalized text is lookup-only and must not replace source text for final spans.
- `EntityAnnotation.validate_offsets(source_text)` must pass for exported predictions when source
  text is available.
- Preprocessing may create mapped text, but every emitted span must map back to the original text.

## Dictionary Safety

- Output codes must exist in the loaded dictionary.
- Entity type and code system must be compatible.
- `DRUG` can map to `RxNorm` or `NONE`, never ICD-10.
- `DISEASE` can map to ICD-10, UMLS, SNOMED, or `NONE`, never RxNorm.
- `SYMPTOM` can map to UMLS, SNOMED, LOCAL, or `NONE`.
- `LAB_RESULT` can map to LOCAL or `NONE`.

## Context Safety

- `NEGATED` disease mentions are not confirmed patient conditions.
- `FAMILY` disease mentions are family-history conditions, not patient-present conditions.
- `HISTORICAL`, `POSSIBLE`, `PLANNED`, and `RESOLVED` must remain distinct from `PRESENT`.
- Section and sentence scope should be used conservatively when applying context cues.

## Relation Safety

- `TREATS` requires a drug head and disease or symptom tail.
- `HAS_SYMPTOM` requires a disease head and symptom tail.
- `HAS_TEST` requires a disease or finding head and lab test tail.
- `HAS_VALUE` requires a lab test head and lab result tail.
- `HAS_DOSE` requires a drug head and lab result tail.
- `SUGGESTS` requires a lab test or finding head and disease or finding tail.
- Relation types not explicitly allowed by KG constraints must be rejected.

## Validation Gate

Use `scripts/validate_predictions.py` to check schema, offsets, dictionary codes, and KG relation
constraints before treating exported JSONL as valid output.
