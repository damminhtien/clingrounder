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

## Command Output Hygiene

Use context engineering first: keep tasks small, use technical memory selectively, search with `rg`,
run targeted tests in the edit loop, and summarize after each phase.

When `rtk` is available, prefer it for commands whose output can get noisy:

```bash
rtk pytest tests/
rtk ruff check .
rtk mypy src
rtk git status
rtk git diff --stat
```

Use raw commands instead of `rtk` when exact output matters:

- prediction JSONL that must be inspected line by line;
- medical dictionary samples or other source data;
- benchmark raw logs that must be preserved;
- hard-to-reproduce tracebacks where full stack details matter;
- any command where truncation/filtering could hide the evidence needed for the task.

RTK is optional. If it is unavailable or obscures useful details, run the underlying command
directly and save verbose output to a file when needed.

## Task Workflow

1. Read this file first.
2. Treat `.cursor/rules/` as always-on guardrails for project structure, medical NLP invariants,
   and verification.
3. Load only the minimal technical memory needed for the task:
   - default: `docs/invariants.md`, `docs/schema.md`, the module under change, and nearby tests;
   - architecture/stack decisions: add `docs/architecture.md` and relevant files in `docs/decisions/`;
   - metrics/experiments: add `docs/evaluation.md`;
   - broad design questions: add `docs/design.md`.
4. For module-specific work, read the matching `.claude/skills/*/SKILL.md` file as the local
   playbook before editing:
   - schema/export/metrics: `schema-evaluator`
   - offsets/preprocessing/spans: `offset-safety`
   - dictionaries/retrieval/linking: `dictionary-linker`
   - assertions/context: `context-reasoning`
   - KG/relation constraints: `kg-constraints`
   - experiments/ablations: `experiment-runner`
   - profiling/performance: `performance-benchmark`
   - reviews: `code-reviewer`
5. Use `rg` before opening long files. Examples:
   - `rg "AssertionStatus" src tests`
   - `rg "normalize_for_match|normalize_mention" src tests`
   - `rg "class .*Entity|EntityAnnotation" src tests`
6. Inspect only the modules and tests needed for the task. Do not read the whole repo unless the
   task is explicitly cross-cutting.
7. Make a short plan for non-trivial changes.
8. Keep edits scoped to the requested module.
9. Add or update tests before final verification.
10. Run targeted tests during iteration, for example:
    - `.venv/bin/python -m pytest tests/test_context_rules.py -q`
    - `.venv/bin/python -m pytest tests/test_offset_mapping.py -q`
    - `.venv/bin/python -m pytest tests/test_candidate_generation.py -q`
11. Run full verification before handoff when feasible.
12. Update docs when behavior or commands change.

## Definition of Done

A task is done only when implementation is typed, tests pass or failures are reported, offset
regression is checked when spans are touched, JSON schema validation still works, and known
limitations are documented.
