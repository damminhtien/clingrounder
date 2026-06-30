# Schema

Internal schemas live under `src/medical_kg_nlp/schema/` and use typed dataclasses plus enums.

## Core Types

- `ClinicalDocument`: `document_id`, source `text`, and string metadata.
- `EntityAnnotation`: stable `id`, source `span`, source `text`, normalized text, entity type,
  assertion, code system, code, confidence, and candidate list.
- `CandidateConcept`: dictionary candidate metadata for debugging and recall evaluation.
- `RelationAnnotation`: typed edge between entity ids with optional evidence span.
- `ClinicalPrediction`: exported prediction object with `document_id`, `text_hash`, entities,
  relations, and metadata.

## Enum Sets

- `EntityType`: disease, symptom, drug, lab test, lab result, procedure, patient info, anatomy,
  finding, other.
- `AssertionStatus`: present, negated, historical, family, possible, conditional, planned,
  resolved, unknown.
- `CodeSystem`: ICD-10, RxNorm, UMLS, SNOMED, LOCAL, NONE.
- `RelationType`: treatment, symptom, test/value/dose links, causal/associative links, ontology
  edges, and unknown.

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
- relation endpoint existence and type compatibility.

Command:

```bash
python scripts/validate_predictions.py \
  --pred outputs/predictions.jsonl \
  --documents data/samples/sample_notes.jsonl \
  --dictionary data/dictionaries/seed_concepts.jsonl
```
