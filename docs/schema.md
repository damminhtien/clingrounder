# Schema

Internal schemas live under `src/medical_kg_nlp/schema/` and use typed dataclasses plus enums.

## Core Types

- `ClinicalDocument`: `document_id`, source `text`, and string metadata.
- `EntityAnnotation`: stable `id`, source `span`, source `text`, normalized text, entity type,
  primary assertion, multi-label assertion features, code system, code, confidence, and candidate
  list. `assertion_evidence` records the rule id, cue, assertion, and scope behind each decision.
- `MedicationMention`: original drug span, validated full medication span, and typed component spans
  for administered dose, form, release, route, frequency, duration, transition, and context text.
  Product strength remains dictionary metadata and is not conflated with administered dose. Phase 1 export reads
  this structure and does not apply a second regex span patch.
- `CandidateConcept`: dictionary candidate metadata for debugging and selective export. It records
  independent `retrieval_score` and `emit_probability` values, the primary `source`, all
  `evidence_sources`, `matched_alias`, `qualified`, and `qualification_reason`. Unqualified
  candidates remain available in traces but are not eligible for Phase 1 export. No confidence is
  inferred from exact matching: sources without an explicit calibrated probability carry `0.0`.
- `RelationAnnotation`: typed edge between entity ids with optional evidence span.
- `ClinicalPrediction`: exported prediction object with `document_id`, `text_hash`, entities,
  relations, and metadata.

## Enum Sets

- `EntityType`: disease, symptom, drug, lab test, lab result, dosage, strength, frequency, route,
  duration, dosage form, procedure, patient info, anatomy, finding, other.
- `AssertionStatus`: present, negated, historical, family, possible, conditional, planned,
  resolved, unknown.
- `CodeSystem`: ICD-10, RxNorm, UMLS, SNOMED, LOCAL, NONE.
- `RelationType`: treatment, symptom, test/value links, medication dose/route/frequency/duration/
  dosage-form links, causal/associative links, ontology edges, and unknown.

## JSON Validation

`PredictionValidator` parses prediction JSON into internal dataclasses and reports structured
issues instead of silently accepting invalid output. It checks:

- required fields and enum values;
- confidence ranges;
- duplicate entity and relation ids;
- source-text offsets and text hash when document text is supplied;
- entity/code-system compatibility;
- entity and candidate dictionary membership when a dictionary is supplied;
- candidate code-system compatibility with the parent entity type;
- complete candidate retrieval/emission provenance and assertion evidence;
- structured medication spans and component kinds when present;
- relation endpoint existence and type compatibility.

Command:

```bash
uv run medical-kg validate \
  --profile development \
  --pred outputs/predictions.jsonl \
  --documents data/samples/sample_notes.jsonl \
  --dictionary data/dictionaries/seed_concepts.jsonl
```

The internal prediction schema is intentionally strict and version-forward. Missing
`assertion_evidence`, legacy candidate `score`, implicit qualification, or missing source provenance
is a schema error rather than a compatibility fallback.

Phase 1 external JSON has a narrower conditional contract. `text`, `type`, `assertions`, and
`position` are always required. `candidates` is required for `THUỐC` and `CHẨN_ĐOÁN`; for the
other three types it may be omitted or emitted as an empty list. Source TXT is decoded from raw
bytes with UTF-8 BOM handling and without universal-newline translation, so CRLF offsets remain
stable through inference, validation, ZIP validation, and hashing.

Medication normalization uses two vocabularies in full mode: a small recognition dictionary for
NER and a larger normalization dictionary for candidate retrieval and validation. Official BTC
sample mappings are stored as provenance-bearing exact mention memory and are ignored unless the
target RxCUI exists in the loaded normalization dictionary.
