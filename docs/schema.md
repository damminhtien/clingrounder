# Schema

Internal schemas live under `src/clingrounder/schema/` and use typed dataclasses plus enums.

## Core Types

- `ClinicalDocument`: `document_id`, source `text`, and string metadata.
- `EntityAnnotation`: stable `id`, source `span`, source `text`, normalized text, entity type,
  primary assertion, multi-label assertion features, code system, code, confidence, and candidate
  list. `assertion_evidence` records the rule id, cue, assertion, and scope behind each decision.
- `MedicationMention`: original drug span, validated full medication span, and typed component spans
  for administered dose, form, release, route, frequency, duration, transition, and context text.
  Product strength remains dictionary metadata and is not conflated with administered dose.
  Task exporters read this structure and must not apply a second regex span patch.
- `CandidateConcept`: dictionary candidate metadata for debugging and selective export. It records
  independent `retrieval_score` and `emit_probability` values, the primary `source`, all
  `evidence_sources`, `matched_alias`, `qualified`, and `qualification_reason`. Unqualified
  candidates remain available in traces but are not eligible for final assignment. No confidence is
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
- assigned entity membership in the active terminology release;
- qualified candidate membership, with explicit debug-only handling for unknown unqualified
  candidates;
- consistent `(code_system, code)` presence for entities and candidates;
- candidate code-system compatibility with the parent entity type;
- complete candidate retrieval/emission provenance and assertion evidence;
- structured medication spans and component kinds when present;
- relation endpoint existence and type compatibility.

Command:

```bash
uv run clingrounder validate \
  --profile development \
  --pred outputs/predictions.jsonl \
  --documents data/samples/sample_notes.jsonl \
  --dictionary data/dictionaries/seed_concepts.jsonl
```

The internal prediction schema is intentionally strict and version-forward. Missing
`assertion_evidence`, legacy candidate `score`, implicit qualification, or missing source provenance
is a schema error rather than a compatibility fallback.

External task schemas belong to benchmark adapters. The archived Phase 1 JSON contract and its
CRLF-sensitive export rules are documented under
[`docs/benchmarks/phase1`](benchmarks/phase1/README.md).

Medication normalization can use a small recognition dictionary for NER and a larger terminology
repository for candidate retrieval and validation. Reviewed exact mention memory is optional,
provenance-bearing, and ignored unless its target exists in the active repository.
