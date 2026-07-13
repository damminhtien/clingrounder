# Ontological Reasoning in Medical Knowledge Retrieval

Prototype system for clinical entity extraction, dictionary-constrained normalization, context reasoning, relation extraction, and lightweight KG validation.

The internal schema stays rich for debugging, while the Phase 1 exporter writes the official flat
entity JSON files for `input/1.txt` through `input/100.txt`.

Supported runtime: Python 3.11–3.14, matching `pyproject.toml` and the CI matrix.

## What Works Now

- Typed internal schema for documents, entities, candidates, relations, and predictions.
- Offset-preserving preprocessing utilities.
- Structured seed ICD-10, RxNorm, Vietnamese alias, abbreviation, and local concept dictionaries.
- Exact, fuzzy, abbreviation, character n-gram, and BM25-style candidate generation.
- Rule-based NER over dictionary aliases plus lab/dose regexes.
- Assertion rules for present, negated, historical, family, possible, planned, and resolved contexts.
- Relation rules for treatment, symptom, test suggestion, and dose links.
- KG constraints that prevent invalid code-system and relation-type outputs.
- Phase 1 flat JSON exporter, validator, ZIP builder, and stage-wise Phase 1 metrics.
- JSONL pipeline, data profiler, evaluator, ablation timing reports, error analysis CSV, and pytest coverage.

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
uv run python scripts/profile_data.py --documents data/samples/sample_notes.jsonl --gold data/samples/gold.jsonl --output outputs/profiles/sample_profile.json --markdown outputs/profiles/sample_profile.md
uv run python scripts/build_phase1_submission.py --input-dir data/raw/input --run-root outputs/runs --output-dir phase1/output --zip phase1/output.zip --expected-count 100
uv run python scripts/run_ablation.py --config configs/ablations.yaml --run-root outputs/runs
uv run pytest tests/
```

The default Phase 1 config uses the validated entity-only policy: entities are exported while
`assertions` and `candidates` remain empty, and context/linking/KG stages that cannot affect the
submission are not constructed. Use `--config configs/phase1_full.yaml` for a controlled
assertion/candidate diagnostic. `configs/phase1_selective.yaml` enables evidence-gated assertions
while retaining candidate abstention. `configs/phase1_selective_candidates.yaml` is a local
calibration probe for reviewed `dictionary_exact` mappings; it is not promoted until a public probe
wins.

Internal predictions use the current strict schema only. Candidate rows require separate
`retrieval_score` and `emit_probability` values plus source/evidence provenance; assertion rows
require `assertion_evidence`. Legacy prediction JSON is intentionally rejected. Dictionary-pinned
candidates do not receive an implicit confidence: an unconfigured `(code system, source)` has
`emit_probability: 0` and therefore abstains under selective export.

For controlled Top 10 experiments, use `scripts/run_phase1_top10_probes.py`. It builds
content-hashed, strict-validated ZIPs for lab precision, strict exclusions, overlap resolution,
selective assertions, and reviewed candidate tiers while keeping holdout sealed by default. See
[`docs/evaluation.md`](docs/evaluation.md#top-10-probe-suite).

Evaluate a flat output directory against the reviewed manual-gold split:

```bash
uv run python scripts/evaluate_phase1_manual_gold.py \
  --gold-dir data/manual_gold \
  --pred-dir outputs/phase1/<run>/phase1/output \
  --output-dir outputs/evaluation/manual_gold
```

Compile the reviewed exact-unique whitelist and run document-fold candidate calibration:

```bash
uv run python scripts/build_phase1_reviewed_candidates.py --split train
uv run python scripts/build_phase1_submission.py --config configs/phase1_full.yaml
uv run python scripts/calibrate_phase1_candidates.py \
  --pred outputs/phase1/<full-run>/phase1/full_internal_predictions.jsonl \
  --reviewed-map data/manual_gold/reviewed_candidate_map.jsonl \
  --output-dir outputs/phase1/<full-run>/calibration
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
python scripts/profile_data.py --documents data/samples/sample_notes.jsonl --gold data/samples/gold.jsonl --output outputs/profiles/sample_profile.json --markdown outputs/profiles/sample_profile.md
python scripts/build_phase1_submission.py --input-dir data/raw/input --run-root outputs/runs --output-dir phase1/output --zip phase1/output.zip --expected-count 100
python scripts/run_ablation.py --config configs/ablations.yaml --run-root outputs/runs
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

Note: PyTorch and Accelerate wheels are platform-specific. On Python 3.14, they are enabled where
compatible wheels exist; macOS x86_64 may need an official PyTorch index, a different Python build,
or a source install.

## Hashed Run Outputs

Use `--run-root outputs/runs` on long-running commands to avoid overwriting old results. Each run
creates a directory like `outputs/runs/20260702T143000Z_phase1_a1b2c3d4e5/` with a
`run_manifest.json`; relative output paths are written inside that directory. The timestamp keeps
runs unique, while the digest and manifest record content hashes, Git state, resolved config,
Python version, dependency lock hash, random seed, and command.

```bash
uv run python scripts/build_phase1_submission.py \
  --input-dir data/raw/input \
  --run-root outputs/runs \
  --output-dir phase1/output \
  --zip phase1/output.zip \
  --expected-count 100
```

## Expected Pipeline

```text
Raw clinical text
  -> document loading
  -> offset-preserving preprocessing
  -> section detection
  -> sentence splitting
  -> entity extraction
  -> context/assertion classification
  -> candidate generation
  -> candidate reranking
  -> normalization assignment
  -> ICD/RxNorm/UMLS validation
  -> relation extraction (internal reasoning for Phase 1)
  -> ontology/KG consistency checking
  -> structured JSON output
  -> Phase 1 flat JSON/ZIP export when building a submission
  -> prediction validation
  -> evaluation and error analysis
```

`configs/phase1_submission.yaml` stops after entity extraction for official entity-only runs.
`configs/phase1_full.yaml` enables context, candidate fusion, medication-aware reranking,
separates the compact recognition dictionary from the RxNorm Full normalization store,
confidence-margin qualification, and KG validation, and saves the internal prediction JSONL needed
for calibration plus per-document trace JSONL. Only a reviewed, cross-fitted selective policy may
treat retrieval output as submission-ready.

## Repository Layout

```text
configs/                  YAML configuration
data/dictionaries/         Seed ICD/RxNorm/local dictionaries
data/samples/              Synthetic notes and gold annotations
docs/                      Architecture, dictionaries, invariants, schema, evaluation, decisions
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
- Local shortcuts: `make lint`, `make type`, `make test`, `make pipeline`, `make validate`,
  `make evaluate`, `make profile`, `make phase1-submit`, `make phase1-validate`, `make ablation`.

## Current Limitations

- Runtime dictionaries are conservative reviewed subsets; full TT06 and versioned RxNorm source layers remain separate from runtime promotion.
- Transformer NER, context, and relation classifiers are placeholders.
- Public dataset adapters are schema-compatible placeholders until local dataset paths are supplied.
- Hidden Phase 1 test data has no gold labels, so official-style `phase1_score` is only local on
  synthetic or labeled regression data; test submissions are gated by schema, offset, dictionary,
  and ZIP validation.

The current priority is correctness of schema, offsets, linking constraints, context handling, and end-to-end debuggability before adding large models.
