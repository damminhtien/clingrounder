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
- CLI: installed `medical-kg` command using standard-library `argparse`; FastAPI/Uvicorn only when
  serving is needed.
- Experiments: Hydra/OmegaConf and MLflow optional; W&B separate.

## Pipeline

```text
Raw clinical text
  -> document_loader
  -> lookup_normalization_diagnostics
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

Medication-list parsing is structural rather than vocabulary-specific. It recognizes numbered,
parenthesized, bulleted, and inline items, separates an indication clause from the medication SIG,
and extends validated drug spans through contiguous attributes. It does not delete or recreate
indication entities already emitted by the configured NER stage. The organizer's executable
example is isolated under the Phase 1 benchmark resources; its recognition vocabulary and reviewed
RxNorm mappings are loaded only when a benchmark explicitly opts in. Core parsers never load that
fixture or materialize indication entities from hidden global vocabulary.
When the dictionary genuinely provides both disease and symptom entries for the same indication
span, the indication context selects symptom; diagnosis-only concepts remain diagnoses. Outside a
medication indication, the conservative disease fallback remains until a model-backed type resolver
is calibrated.

`lookup_normalization_diagnostics` records the lookup-normalization size and offset map; it is not a
preprocessing input to later stages. Downstream NER, context, linking, and relation stages consume
original source text. Normalized text should become a span-producing input only after every output
is mapped back to source offsets with end-to-end regression tests.

Section detection is configuration-driven. `SectionRuleRegistry` owns heading aliases, semantic
categories, optional parent constraints, and scope limits; `RuleBasedSectionDetector` returns
source-coordinate section and heading spans. Downstream policy should depend on the semantic
category rather than one literal Vietnamese heading. Third-party sentence or word tokenization
cannot own exported offsets.

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

Local Hugging Face adapters implement fast-tokenizer NER projection, text encoding, and
cross-encoder reranking. They lazy-load the `ml` extra, require `model_id` plus a pinned `revision`,
and pass `local_files_only=true`; importing the core pipeline never imports PyTorch or Transformers.
Dense retrieval consumes a separate `DenseVectorIndexPort`, so adding an ANN backend cannot bypass
terminology type/code-system constraints. No dense backend is enabled by default. The correctness
gate remains candidate recall at 20 plus a model-revision-specific latency/RSS benchmark before an
expensive retriever or reranker is promoted.

Generated candidates remain in internal predictions for recall/rank analysis. Only candidates with
`qualified=true` are eligible for final assignment or task export. `retrieval_score` ranks candidates;
`emit_probability` is a separate calibrated decision value. The primary source and all fusion
evidence sources are stored separately. A benchmark adapter may apply a task-specific threshold
matrix or reviewed whitelist after qualification without changing reusable retrieval behavior.

Reviewed mention memory is optional and fail-closed. Each row is keyed by normalized mention plus
entity type and must carry reviewed status, source SHA-256, and compatible terminology releases.
Medication rows also compare parsed strength/form/route structure before a terminal match. The
runtime candidate source remains the stable value `reviewed_memory`; artifact provenance belongs to
the memory contract rather than becoming a new calibration source name.

RxNorm qualification receives the validated full medication span. Medication structure keeps
ingredient/brand, product strength, administered dose, dosage form, route, frequency, and release
type as distinct concepts. In particular, `po` is a route rather than evidence for a tablet and
`iv` is a route rather than evidence for an injection product. A strength mismatch hard-rejects a
candidate only when the mention provides product evidence such as amount plus explicit form or
release type. Dose ranges and route/frequency SIG amounts remain administered or ambiguous doses
and can affect ranking without rejecting a valid product. Explicit product-strength, release-type,
or dose-form conflicts retain a structured rejection reason; rejected candidates remain in the
internal list for error analysis rather than being deleted from retrieval traces.

Dictionary aliases with unresolved cross-type semantics are retained as non-exportable
`AmbiguousEntityProposal` records. Rule-only extraction abstains. A hybrid extractor may use an
exact-span proposal as a small support bonus only when an independent model predicts one of its
candidate types; the proposal can never become a final entity by itself.

## Assertion Rules

`data/heuristics/assertion_cues.jsonl` is the source of truth for cue text, direction, section priors,
provenance, priority, and scope distance. Every loaded cue receives a stable `rule_id` and the same
resource is packaged under `src/medical_kg_nlp/resources/`. The classifier returns rule evidence and
`PipelineTrace` counts matched rule IDs, while Python owns only generic scope execution and explicit
false-positive mechanics. Cue inventories must not be duplicated as fallback lists in Python.
Within a direction, executable cue matches are ordered by descending rule `priority`, then nearest
distance, longer cue, and stable `rule_id`. A match beyond that rule's `max_distance` is discarded.
This ordering is independent of JSONL row order; evidence records point to the exact rule selected
by execution rather than performing a second metadata lookup after matching.

Batch assertion classification also emits a modifier-target `ContextGraph`. Cue occurrences are
raw-coordinate modifier nodes, entities are target nodes, and edges retain assertion type, scope,
distance, and stable rule provenance. Section priors are explicit spanless modifier nodes. The
graph supports model features and error analysis without changing deterministic assertion policy.
`AssertionModelFeatureExtractor` converts this evidence into bounded sparse features while keeping
negation, history, family, uncertainty, conditional, planned, and resolved status independent.

Dependency paths may later add evidence to this graph, but cannot replace linear evidence until a
Vietnamese parser passes target-anchored assertion and raw-offset benchmarks. Likewise,
word-segmented Vietnamese encoders may supply representations but cannot own final NER boundaries
until segmentation is reversibly projected to source text. The detailed source audit and adoption
map are documented in `docs/reference-implementations.md`.

## Data Storage

Canonical terminology stays in JSONL. Full lexical lookup uses a derived SQLite FTS5 index keyed by
source, schema, and normalization fingerprints; runtime connections are read-only and thread-local.
Prototype tabular data remains Parquet/DuckDB-compatible. A graph database is deferred until
interactive graph traversal becomes a real requirement.

## Agentic Workflow

Agents should read `AGENTS.md`, then the smallest relevant docs and modules. Work should be split by
module boundaries such as schema, offset mapping, context rules, dictionary linking, or KG
constraints. Large cross-repo refactors are out of scope unless explicitly requested.
