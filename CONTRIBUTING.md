# Contributing

Thanks for improving this prototype. The project is a Python-first clinical NLP and medical KG
retrieval baseline, so correctness and reproducibility matter more than broad refactors.

## Ground Rules

- Read `AGENTS.md` before making changes.
- Preserve original character offsets.
- Keep final codes dictionary-constrained.
- Do not map drugs to ICD-10 or diseases to RxNorm.
- Keep negated and family-history conditions distinct from present patient conditions.
- Keep changes scoped to the relevant module and tests.

## Development Setup

Preferred:

```bash
uv sync --extra dev
uv run pre-commit install
```

Fallback:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pre-commit install
```

## Workflow

1. Create a small branch for the task.
2. Use `rg` to find relevant code and tests before opening long files.
3. Add focused tests for behavior changes.
4. Run targeted tests while iterating.
5. Run full verification before opening a PR when feasible.

Recommended checks:

```bash
ruff check .
mypy src
pytest tests
uv run medical-kg validate \
  --profile development \
  --pred outputs/predictions.jsonl \
  --documents data/samples/sample_notes.jsonl \
  --dictionary data/dictionaries/seed_concepts.jsonl
```

Use `python -m medical_kg_nlp.cli ...` if the console script is not installed in the active venv.

## Pull Requests

PRs should include:

- What changed and why.
- Tests and validation run.
- Any offset, schema, dictionary, context, or KG risk.
- Known limitations or follow-up work.

Do not include private clinical data, licensed dataset files, access tokens, or large generated
artifacts.
