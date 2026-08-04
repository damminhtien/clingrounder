# Invariants

## Offset Safety

- Entity spans are always offsets into the original source text.
- Normalized text is lookup-only and must not replace source text for final spans.
- `EntityAnnotation.validate_offsets(source_text)` must pass for exported predictions when source
  text is available.
- Preprocessing may create mapped text, but every emitted span must map back to the original text.
- The active normalization contract is versioned. Changing normalization semantics requires a
  version change, offset-map regression tests, and rebuilding indexes whose keys use that contract.
- Until all span-producing stages consume mapped text explicitly, normalized text remains
  lookup/diagnostic-only and pipeline stages must receive the original source text.

## Dictionary Safety

- Output codes must exist in the loaded dictionary.
- Assigned codes are valid only as `(CodeSystem.NONE, null)` or `(non-NONE, non-null)` pairs.
- The final pipeline gate proves every assigned code against the active terminology release through
  `TerminologyMembershipPort`; entity type compatibility alone is insufficient.
- Qualified candidates must belong to the active terminology. Unknown unqualified candidates are
  debug-only and require an explicit validator policy; release validation rejects them.
- Entity type and code system must be compatible.
- `DRUG` can map to `RxNorm` or `NONE`, never ICD-10.
- `DISEASE` can map to ICD-10, MONDO, UMLS, SNOMED, or `NONE`, never RxNorm or HPO.
- `SYMPTOM` and `FINDING` can map to HPO, UMLS, SNOMED, LOCAL, or `NONE`.
- MONDO and HPO are opt-in ontology systems; importing their English labels does not make those
  labels eligible for Vietnamese runtime recognition without a separate benchmarked alias policy.
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
- `HAS_DOSE` requires a drug head and dosage or strength tail.
- `HAS_ROUTE` requires a drug head and route tail.
- `HAS_FREQUENCY` requires a drug head and frequency tail.
- `HAS_DURATION` requires a drug head and duration tail.
- `HAS_DOSAGE_FORM` requires a drug head and dosage-form tail.
- `SUGGESTS` requires a lab test or finding head and disease or finding tail.
- Relation types not explicitly allowed by KG constraints must be rejected.

## Validation Gate

Use `medical-kg validate` to check schema, offsets, dictionary codes, and KG relation constraints
before treating exported JSONL as valid output. Runtime uses the `core` profile; ordinary CLI checks
use `development`; submission and artifact gates use `release`. A release check containing assigned
codes must receive a terminology source, otherwise membership cannot be proved and validation fails.
