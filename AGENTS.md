# AGENTS.md

## Project Goal

Build a modular clinical NLP system for medical entity extraction, dictionary-constrained
normalization, context reasoning, relation extraction, and lightweight ontology/KG validation.

The core implementation is Python-first. Rust or C++ should only be introduced after profiling
shows a concrete bottleneck.

## Non-Negotiable Invariants

- Never destroy or rewrite original character offsets.
- Never output medical codes that are absent from the loaded dictionary.
- Never map `DRUG` entities to ICD-10 disease codes.
- Never map `DISEASE` entities to RxNorm drug codes.
- Negated diseases must not be treated as confirmed patient conditions.
- Family-history diseases must not be treated as patient-present diseases.
- Candidate generation must filter by entity type before final linking.
- Every behavior change needs focused tests.

## Architecture Constraints

- Keep the pipeline modular: preprocessing, NER, context, retrieval, linking, relations, KG, and
  evaluation stay behind their existing interfaces.
- Use deterministic rule baselines first; transformer NER, dense retrieval, and rerankers remain
  replaceable extension points.
- Do not add external APIs or hosted services to the core path.
- Do not add Java as a core runtime.
- Do not introduce Neo4j until table-backed graph storage is demonstrably insufficient.
- Add Rust/C++ extensions only with benchmark evidence and a Python fallback.

## Commands

Install with uv when available:

```bash
uv sync --extra dev
```

Fallback install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Lint:

```bash
uv run ruff check .
```

Type check:

```bash
uv run mypy src
```

Test:

```bash
uv run pytest tests
```

Run the sample pipeline:

```bash
uv run python scripts/run_pipeline.py \
  --input data/samples/sample_notes.jsonl \
  --output outputs/predictions.jsonl
```

Validate predictions:

```bash
uv run python scripts/validate_predictions.py \
  --pred outputs/predictions.jsonl \
  --documents data/samples/sample_notes.jsonl \
  --dictionary data/dictionaries/seed_concepts.jsonl
```

If `uv` is not installed, use the same commands through `python -m` or the script path after
installing the dev dependencies.

## Task Workflow

1. Read this file and the relevant docs under `docs/`.
2. Treat `.cursor/rules/` as always-on guardrails for project structure, medical NLP invariants,
   and verification.
3. For module-specific work, read the matching `.claude/skills/*/SKILL.md` file as the local
   playbook before editing:
   - schema/export/metrics: `schema-evaluator`
   - offsets/preprocessing/spans: `offset-safety`
   - dictionaries/retrieval/linking: `dictionary-linker`
   - assertions/context: `context-reasoning`
   - KG/relation constraints: `kg-constraints`
   - experiments/ablations: `experiment-runner`
   - profiling/performance: `performance-benchmark`
   - reviews: `code-reviewer`
4. Inspect only the modules and tests needed for the task.
5. Make a short plan for non-trivial changes.
6. Keep edits scoped to the requested module.
7. Add or update tests before final verification.
8. Run targeted tests during iteration and full tests before handoff when feasible.
9. Update docs when behavior or commands change.

## Definition of Done

A task is done only when implementation is typed, tests pass or failures are reported, offset
regression is checked when spans are touched, JSON schema validation still works, and known
limitations are documented.
