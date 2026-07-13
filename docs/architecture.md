# Architecture

## Decision

The project uses a Python-first core:

```text
Python pipeline
  + deterministic rule baselines
  + dictionary-constrained retrieval
  + lightweight KG validation
  + optional model-backed extension points
```

Rust or C++ is reserved for measured bottlenecks such as phrase matching, fuzzy search, offset
mapping, or candidate merging. Java is not used as a core language because the biomedical NLP,
retrieval, and experiment-analysis ecosystem is stronger in Python for this project phase.

## Stack

- Language: Python 3.11–3.14; CI verifies every declared minor version.
- Package/project: `uv` with `pyproject.toml` as the source of truth.
- Code quality: ruff, mypy, pytest, pre-commit.
- Schema: Pydantic v2 is available for external schemas; internal baseline schemas use dataclasses
  and enums.
- Data: Polars, DuckDB, PyArrow/Parquet, and JSON Lines are optional extras for larger datasets.
- NLP/ML: PyTorch, Transformers, Datasets, Tokenizers, Accelerate, and scikit-learn are optional
  model extras. PyTorch and Accelerate are platform-specific; keep markers where Python 3.14 wheels
  are not available for a local platform.
- Retrieval: built-in exact/fuzzy/char-ngram/BM25 baseline, with optional bm25s, RapidFuzz,
  FAISS CPU, and Qdrant client.
- Graph: lightweight in-memory graph first, with optional NetworkX and DuckDB/SQLite-ready tables.
- API/CLI: Typer/Rich for future CLIs; FastAPI/Uvicorn only when serving is needed.
- Experiments: Hydra/OmegaConf and MLflow optional; W&B separate.

## Pipeline

```text
Raw clinical text
  -> document_loader
  -> offset_preserving_preprocessing
  -> section_detection
  -> sentence_splitting
  -> entity_extraction
     -> structured medication mention parsing
  -> context_assertion_classification
  -> candidate_generation
  -> candidate_reranking
  -> normalization_assignment
  -> icd_rxnorm_umls_validation
  -> relation extraction
  -> ontology_kg_consistency_check
  -> structured_json_output
  -> prediction_validation
  -> evaluation and error analysis
```

Each runner stage emits timing and counters through `PipelineTrace`. Stage names are intentionally
stable so ablations can compare runtime, validation issues, linking behavior, context behavior, and
relation quality while swapping one component at a time.

The current deterministic baseline keeps `offset_preserving_preprocessing` diagnostic-only:
downstream NER, context, linking, and relation stages consume original source text. Normalized text
should only feed downstream stages after normalized spans are mapped back to source offsets end to
end with offset regression tests.

## Retrieval

Candidate generation must avoid brute-force mention-to-dictionary comparison. The intended order is:

```text
mention
  -> exact lookup
  -> return immediately when exact/type-compatible output has one unique code
  -> abbreviation lookup
  -> fuzzy top-k
  -> character n-gram top-k
  -> BM25 top-k
  -> optional dense top-k
  -> merge and deduplicate
  -> type/code-system filter
  -> rerank top-k only
  -> qualify by absolute threshold and relative top-score margin
  -> export dynamic top-k, capped at 5
```

Dense retrieval and cross-encoder reranking are extension points. The correctness gate is
candidate recall at 20 before expensive reranking is added.

Generated candidates remain in internal predictions for recall/rank analysis. Only candidates with
`qualified=true` are eligible for Phase 1 export. `retrieval_score` ranks candidates;
`emit_probability` is a separate calibrated decision value. The primary source and all fusion
evidence sources are stored separately. Selective Phase 1 export applies a `(code system, primary
source)` threshold matrix and an exact reviewed whitelist after qualification.

RxNorm qualification receives the validated full medication span. Explicit strength, release type,
or dose-form conflicts hard-reject a candidate with a structured reason; the candidate remains in
the internal list for error analysis rather than being deleted from retrieval traces.

## Assertion Rules

`data/heuristics/assertion_cues.jsonl` is the source of truth for cue text, direction, section priors,
provenance, priority, and scope distance. Every loaded cue receives a stable `rule_id` and the same
resource is packaged under `src/medical_kg_nlp/resources/`. The classifier returns rule evidence and
`PipelineTrace` counts matched rule IDs, while Python owns only generic scope execution and explicit
false-positive mechanics. Cue inventories must not be duplicated as fallback lists in Python.

## Data Storage

Prototype data stays in JSONL, Parquet-compatible tables, DuckDB/SQLite-ready graph tables, and
small local indexes. A graph database is deferred until interactive graph traversal becomes a real
requirement.

## Agentic Workflow

Agents should read `AGENTS.md`, then the smallest relevant docs and modules. Work should be split by
module boundaries such as schema, offset mapping, context rules, dictionary linking, or KG
constraints. Large cross-repo refactors are out of scope unless explicitly requested.
