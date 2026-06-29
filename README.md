# Ontological Reasoning in Medical Knowledge Retrieval

Prototype system for clinical entity extraction, dictionary-constrained normalization, context reasoning, relation extraction, and lightweight KG validation.

The official competition schema is not available yet, so this repository starts with a small but complete Vietnamese/English synthetic baseline that is easy to adapt when the final data arrives.

## What Works Now

- Typed internal schema for documents, entities, candidates, relations, and predictions.
- Offset-preserving preprocessing utilities.
- Seed ICD-10, RxNorm, and local concept dictionaries.
- Exact, fuzzy, abbreviation, character n-gram, and BM25-style candidate generation.
- Rule-based NER over dictionary aliases plus lab/dose regexes.
- Assertion rules for present, negated, historical, family, possible, planned, and resolved contexts.
- Relation rules for treatment, symptom, test suggestion, and dose links.
- KG constraints that prevent invalid code-system and relation-type outputs.
- JSONL pipeline, evaluator, error analysis CSV, and pytest coverage.

## Quick Start

Preferred:

```bash
uv sync --extra dev
uv run pre-commit install

uv run python scripts/build_dictionaries.py --config configs/default.yaml
uv run python scripts/build_indexes.py --config configs/default.yaml
uv run python scripts/run_pipeline.py --input data/samples/sample_notes.jsonl --output outputs/predictions.jsonl
uv run python scripts/validate_predictions.py --pred outputs/predictions.jsonl --documents data/samples/sample_notes.jsonl --dictionary data/dictionaries/seed_concepts.jsonl
uv run python scripts/evaluate.py --gold data/samples/gold.jsonl --pred outputs/predictions.jsonl
uv run pytest tests/
```

Fallback without `uv`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pre-commit install

python scripts/build_dictionaries.py --config configs/default.yaml
python scripts/build_indexes.py --config configs/default.yaml
python scripts/run_pipeline.py --input data/samples/sample_notes.jsonl --output outputs/predictions.jsonl
python scripts/validate_predictions.py --pred outputs/predictions.jsonl --documents data/samples/sample_notes.jsonl --dictionary data/dictionaries/seed_concepts.jsonl
python scripts/evaluate.py --gold data/samples/gold.jsonl --pred outputs/predictions.jsonl
python -m pytest tests/
```

## Optional Stacks

The baseline installs only lightweight runtime and dev dependencies. Add extras as needed:

```bash
uv sync --extra data          # polars, duckdb, pyarrow, jsonlines
uv sync --extra retrieval     # rapidfuzz, bm25s, faiss-cpu
uv sync --extra graph         # networkx
uv sync --extra ml            # torch, transformers, datasets, tokenizers, accelerate, scikit-learn
uv sync --extra cli           # typer, rich
uv sync --extra api           # fastapi, uvicorn
uv sync --extra experiment    # hydra-core, omegaconf, mlflow
uv sync --extra wandb         # optional W&B tracking
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
docs/                      Architecture, invariants, schema, evaluation, decisions
scripts/                   Pipeline, evaluation, dictionary, and index commands
src/medical_kg_nlp/        Python package
tests/                     Unit and smoke tests
.cursor/rules/             Cursor project rules
.claude/skills/            Claude skill briefs for module-focused agent work
AGENTS.md                  Repo instructions for coding agents
```

## Project Hygiene

- License: MIT, see `LICENSE`.
- Contributions: see `CONTRIBUTING.md`.
- Security and private-data handling: see `SECURITY.md`.
- Conduct: see `CODE_OF_CONDUCT.md`.
- Changelog: see `CHANGELOG.md`.
- CI: GitHub Actions workflow under `.github/workflows/ci.yml`.
- Local shortcuts: `make lint`, `make type`, `make test`, `make pipeline`, `make validate`.

## Current Limitations

- Dictionaries are seed dictionaries, not full ICD-10/RxNorm releases.
- Transformer NER, context, and relation classifiers are placeholders.
- Public dataset adapters are schema-compatible placeholders until local dataset paths are supplied.
- Evaluation is focused on the internal schema and synthetic sample.

The current priority is correctness of schema, offsets, linking constraints, context handling, and end-to-end debuggability before adding large models.
