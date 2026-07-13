# Schema

Internal schemas live under `src/medical_kg_nlp/schema/` and use typed dataclasses plus enums.

## Core Types

- `ClinicalDocument`: `document_id`, source `text`, and string metadata.
- `EntityAnnotation`: stable `id`, source `span`, source `text`, normalized text, entity type,
  primary assertion, multi-label assertion features, code system, code, confidence, and candidate
  list.
- `MedicationMention`: original drug span, validated full medication span, and typed component spans
  for strength, form, route, frequency, duration, transition, and context text. Phase 1 export reads
  this structure and does not apply a second regex span patch.
- `CandidateConcept`: dictionary candidate metadata for debugging and recall evaluation. It records
  `qualified` and `qualification_reason`; unqualified candidates remain available in traces but are
  not eligible for Phase 1 export.
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
- structured medication spans and component kinds when present;
- relation endpoint existence and type compatibility.

Command:

```bash
python scripts/validate_predictions.py \
  --pred outputs/predictions.jsonl \
  --documents data/samples/sample_notes.jsonl \
  --dictionary data/dictionaries/seed_concepts.jsonl
```
