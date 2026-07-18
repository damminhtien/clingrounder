# Code Map

This map describes ownership and stable extension points in `medical-kg-nlp` 0.2. Start here before
searching implementation details.

## Dependency Direction

```text
schema + preprocessing + terminology ports
                 ↓
          pipeline ports
                 ↓
       rule/model adapters
                 ↓
       PipelineComponents
                 ↓
         PipelineRunner

generic evaluation ← task adapters ← benchmarks
experiments may depend on pipeline/evaluation/benchmarks, never the reverse

source registry -> mining connectors -> artifact store -> parsers -> curation -> snapshots
```

`PipelineRunner` does not read files or construct implementations. `PipelineFactory` is the only
composition root that turns config into a runnable component graph.

## Package Ownership

| Package | Owns | Does not own |
| --- | --- | --- |
| `schema/` | Documents, annotations, predictions, strict payload parsing | Task export formats |
| `preprocessing/` | Normalization contracts and raw offset mapping | Entity decisions |
| `pipeline/` | Ports, component container, factory, runner, batch execution | Model internals or CLI parsing |
| `adapters/rules.py` | Rule implementations behind pipeline ports | Pipeline orchestration |
| `adapters/huggingface/` | Local-only token classifier, encoder, and cross-encoder adapters | Hosted inference or downloads |
| `dictionaries/` | Canonical JSONL records and import utilities | Persistent query backend |
| `terminology/` | Storage-neutral repository port and SQLite FTS5 implementation | Entity extraction policy |
| `retrieval/` | Retriever adapters, fusion, dense-index port | Code assignment policy |
| `linking/` | Candidate reranking, qualification, and assignment | Terminology storage |
| `ner/` | Rule extraction and medication/lab span helpers | Pipeline construction |
| `context/` | Assertion scope and cue execution | Competition labels |
| `relations/`, `kg/` | Relation extraction and medical constraints | Task packaging |
| `evaluation/` | Neutral records, matchers, metrics, and report rendering | Phase 1 imports |
| `experiments/` | Ablations, journals, and agent-facing experiment loops | Reusable metrics |
| `benchmarks/phase1/` | Phase 1 schema, scoring, export, and campaign code | Generic evaluation behavior |
| `validation/` | Core/development/release severity and generic artifact checks | Task-specific ZIP layout |
| `mining/` | Licensed acquisition, immutable artifacts, parsers, curation, review, and snapshots | Competition schemas or hosted services |
| `cli/` | `argparse` command routing and thin IO handlers | Metrics or pipeline algorithms |

## Public Ports

The replaceable contracts live in [`pipeline/ports.py`](../src/medical_kg_nlp/pipeline/ports.py):

- `EntityExtractorPort`
- `AssertionClassifierPort`
- `CandidateRetrieverPort`
- `CandidateRerankerPort`
- `CandidateAssignerPort`
- `RelationExtractorPort`
- `KnowledgeValidatorPort`
- `TerminologyRepository`

Inject implementations through `PipelineComponents` for tests or custom applications. Use
`PipelineFactory.from_config()` at application boundaries. Do not add IO or config parsing to
`PipelineRunner`.

## Configuration Keys

`PipelineFactoryConfig.from_mapping()` accepts three top-level blocks:

```yaml
terminology:
  recognition_path: data/dictionaries/seed_concepts.jsonl
  normalization_paths:
    - data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl
    - data/standards/rxnorm/processed/rxnorm_full_07062026_concepts.jsonl
  normalization_index_path: .cache/medical-kg/terminology/full.sqlite3
  normalization_alias_overlay_paths:
    - outputs/mining/knowledge/dailymed-rxnorm-2026-07-17/alias_overlay.jsonl
  cache_dir: .cache/medical-kg/terminology
  additional_recognition_path: null
  abbreviation_path: data/dictionaries/abbreviations.jsonl
  alias_overlay_path: data/dictionaries/vietnamese_medical_alias.jsonl

pipeline:
  version: 0.2.0
  enable_context: true
  enable_linking: true
  enable_candidate_reranking: true
  enable_entity_kg_validation: true
  enable_relations: true
  enable_relation_kg_validation: true
  max_candidates: 20
  candidate_sources: [exact, abbreviation, fuzzy, char_ngram, bm25]

models:
  entity_extractor:
    model_id: /absolute/path/to/local-model
    revision: pinned-revision
    device: cpu
    batch_size: 16
    max_length: 512
    stride: 64
    label_map: {DISEASE: DISEASE, DRUG: DRUG}
  candidate_reranker:
    model_id: /absolute/path/to/local-reranker
    revision: pinned-revision
    model_weight: 0.75
    positive_label_index: 1
```

Model blocks are optional. They lazy-load the `ml` extra and set `local_files_only=true`; core
imports do not import PyTorch or Transformers.

## Terminology Lifecycle

JSONL remains the source of truth. SQLite is a derived, immutable runtime index:

```bash
uv run medical-kg terminology build \
  --source data/processed/full_concepts.jsonl \
  --cache-dir .cache/medical-kg/terminology

uv run medical-kg terminology inspect \
  --index .cache/medical-kg/terminology/<fingerprint>.sqlite3 \
  --query metformin \
  --entity-type DRUG \
  --code-system RxNorm
```

The cache key includes source SHA-256, schema version, and normalization version. Runtime opens
read-only, query-only, thread-local connections and never rebuilds a stale index implicitly.

## CLI And Validation

```text
medical-kg pipeline run
medical-kg terminology build|inspect
medical-kg evaluate
medical-kg validate
medical-kg benchmark phase1
medical-kg data registry validate
medical-kg data source sync
medical-kg data dataset build
medical-kg data label propose
medical-kg data review export|import
medical-kg data coverage report
medical-kg data snapshot freeze
medical-kg data run
```

Validation profiles:

- `core`: hard schema, offset, type/code-system, duplicate-ID, and relation invariants.
- `development`: core errors plus warnings for hash and internal-candidate drift.
- `release`: promotes dictionary/hash issues and deterministic artifact checks to errors.

The runner applies `core` once per document. Exporters apply `release` once per artifact.

## Test Tiers

```bash
# Default unit and contract suite
uv run pytest tests

# Entire public suite
uv run pytest -o addopts='' tests

# One opt-in tier
uv run pytest -o addopts='' -m release tests
```

Markers are `integration`, `release`, `benchmark`, `private`, and `model`. Model tests must use a
local cache and must not download weights.

## Search Recipes

```bash
rg "class .*Port" src/medical_kg_nlp/pipeline src/medical_kg_nlp/terminology
rg "PipelineFactoryConfig|from_mapping" src tests configs
rg "INVARIANT:|SCALING:|MODEL:|LICENSE:|PRIVACY:" src tests
rg "EntityAnnotation|RelationAnnotation" src/medical_kg_nlp/schema tests
rg "TerminologyRepository|exact_lookup|search" src tests
rg "EvaluationAdapter" src/medical_kg_nlp/evaluation src/medical_kg_nlp/benchmarks tests
rg "ValidationProfile" src tests
```

Comments with `INVARIANT:`, `SCALING:`, `MODEL:`, `LICENSE:`, and `PRIVACY:` explain non-obvious
safety, concurrency, projection, source-policy, and data-handling decisions. Ordinary control flow
should remain self-explanatory rather than narrated.
