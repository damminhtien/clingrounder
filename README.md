# Ontological Reasoning in Medical Knowledge Retrieval

Prototype system for clinical entity extraction, dictionary-constrained normalization, context reasoning, relation extraction, and lightweight KG validation.

The official competition schema is not available yet, so this repository starts with a small but complete Vietnamese/English synthetic baseline that is easy to adapt when the final data arrives.

## What Works Now

- Typed internal schema for documents, entities, candidates, relations, and predictions.
- Offset-preserving preprocessing utilities.
- Seed ICD-10, RxNorm, and local concept dictionaries.
- Exact, fuzzy, abbreviation, and BM25-style candidate generation.
- Rule-based NER over dictionary aliases plus lab/dose regexes.
- Assertion rules for present, negated, historical, family, possible, planned, and resolved contexts.
- Relation rules for treatment, symptom, test suggestion, and dose links.
- KG constraints that prevent invalid code-system and relation-type outputs.
- JSONL pipeline, evaluator, error analysis CSV, and pytest coverage.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/build_dictionaries.py --config configs/default.yaml
python scripts/build_indexes.py --config configs/default.yaml
python scripts/run_pipeline.py --input data/samples/sample_notes.jsonl --output outputs/predictions.jsonl
python scripts/evaluate.py --gold data/samples/gold.jsonl --pred outputs/predictions.jsonl
pytest tests/
```

## Expected Pipeline

```text
Raw clinical text
  -> document loading
  -> section and sentence splitting
  -> rule NER
  -> assertion classification
  -> candidate generation and linking
  -> relation extraction
  -> KG validation
  -> structured JSON output
  -> evaluation and error analysis
```

## Repository Layout

```text
configs/                  YAML configuration
data/dictionaries/         Seed ICD/RxNorm/local dictionaries
data/samples/              Synthetic notes and gold annotations
docs/design.md             Technical design
scripts/                   Pipeline, evaluation, dictionary, and index commands
src/medical_kg_nlp/        Python package
tests/                     Unit and smoke tests
```

## Current Limitations

- Dictionaries are seed dictionaries, not full ICD-10/RxNorm releases.
- Transformer NER, context, and relation classifiers are placeholders.
- Public dataset adapters are schema-compatible placeholders until local dataset paths are supplied.
- Evaluation is focused on the internal schema and synthetic sample.

The current priority is correctness of schema, offsets, linking constraints, context handling, and end-to-end debuggability before adding large models.

