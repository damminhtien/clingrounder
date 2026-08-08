# Code Map

This map describes ownership and stable extension points in ClinGrounder 0.1.0a6. Start here before
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
| `adapters/hybrid.py` | Evidence-weighted arbitration of model and dictionary entity proposals | Model loading or task-specific thresholds |
| `dictionaries/` | Canonical JSONL records and import utilities | Persistent query backend |
| `terminology/` | Storage-neutral repository port and SQLite FTS5 implementation | Entity extraction policy |
| `retrieval/` | Retriever adapters, fusion, dense-index port | Code assignment policy |
| `linking/` | Candidate reranking, qualification, and assignment | Terminology storage |
| `ner/` | Rule extraction and medication/lab span helpers | Pipeline construction |
| `context/` | Assertion scope and cue execution | Competition labels |
| `ontology/` | Reusable suppression-rule contracts | Task schemas or code priorities |
| `relations/`, `kg/` | Relation extraction and medical constraints | Task packaging |
| `evaluation/` | Neutral records, matchers, metrics, and report rendering | Benchmark imports |
| `experiments/` | Ablations, journals, and agent-facing experiment loops | Reusable metrics |
| `benchmarks/phase1/` | Archived task schema, ontology, CLI handlers, scoring, export, and campaign code | Generic evaluation behavior |
| `validation/` | Core/development/release severity and generic artifact checks | Task-specific ZIP layout |
| `mining/` | Licensed acquisition, immutable artifacts, parsers, curation, terminology evidence, review, model datasets, and snapshots | Competition schemas or hosted services |
| `training/` | Framework-neutral run contracts, span datasets, model adapters, and release verification | Hidden defaults or model weights bundled in core |
| `governance/` | Artifact fingerprints, model governance metadata, audit, and data policy | Clinical safety certification |
| `cli/` | `argparse` routing and reusable thin IO handlers | Benchmark handlers, metrics, or pipeline algorithms |

## Rule NER

`RuleBasedNER` is a facade over a proposal-first engine. Independent dictionary, medication, lab,
and boundary extractors emit immutable evidence before one global resolver chooses a non-overlapping
set. Context may select only among proposed types, and every final span remains in raw coordinates.

Start with [`rule-ner.md`](rule-ner.md). Useful searches:

```bash
rg "class .*ProposalExtractor|EntityProposal" src/clingrounder/ner tests
rg "type_resolution|boundary_rules|rule_id" src/clingrounder/ner tests
rg "medication_mention|structured_lab" src/clingrounder/ner tests
```

## Public Ports

The replaceable contracts live in [`pipeline/ports.py`](../src/clingrounder/pipeline/ports.py):

- `EntityExtractorPort`
- `AssertionClassifierPort`
- `CandidateRetrieverPort`
- `CandidateRerankerPort`
- `DocumentCandidateRerankerPort`
- `CandidateAssignerPort`
- `RelationExtractorPort`
- `KnowledgeValidatorPort`
- `TerminologyRepository`

Inject implementations through `PipelineComponents` for tests or custom applications. Use
`PipelineFactory.from_config()` at application boundaries. Do not add IO or config parsing to
`PipelineRunner`.

## Configuration Keys

`PipelineConfig.from_mapping()` accepts three top-level blocks:

```yaml
terminology:
  recognition_path: data/dictionaries/seed_concepts.jsonl
  normalization_paths:
    - data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl
    - data/standards/rxnorm/processed/rxnorm_full_07062026_concepts.jsonl
  normalization_index_path: .cache/clingrounder/terminology/full.sqlite3
  normalization_alias_overlay_paths:
    - outputs/mining/knowledge/dailymed-rxnorm-2026-07-17/alias_overlay.jsonl
    - outputs/mining/knowledge/codiesp-icd10-2026-07-18/alias_overlay.jsonl
  cache_dir: .cache/clingrounder/terminology
  additional_recognition_path: null
  abbreviation_path: data/dictionaries/abbreviations.jsonl
  alias_overlay_path: data/dictionaries/vietnamese_medical_alias.jsonl

pipeline:
  version: 0.2.0
  enable_context: true
  enable_linking: true
  enable_candidate_reranking: true
  # Optional second pass. It only reorders retrieved candidates by graph evidence.
  enable_graph_evidence_reranking: false
  graph_evidence_max_bonus: 0.04
  graph_evidence_min_support: 2
  graph_evidence_relation_types: [CO_OCCURS_WITH]
  graph_evidence_cache_size: 4096
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

When graph evidence is enabled, `terminology.knowledge_graph_index_path` is required. The second
pass accepts only exact-unique first-pass links as same-sentence context anchors. It cannot add a
candidate, cross sentence boundaries, or use entities that could not be assigned to a sentence.
Its trace stage is `graph_evidence_reranking`, with counters for anchors, context events, graph
features, and changed top-1 predictions. Keep the option disabled until the graph source and model
NER distribution have passed a held-out benchmark. Node and neighbor caches use a thread-safe LRU;
set `graph_evidence_cache_size` from the expected active concept working set instead of graph size.

Model blocks are optional. They lazy-load the `ml` extra and set `local_files_only=true`; core
imports do not import PyTorch or Transformers.

## Terminology Lifecycle

JSONL remains the source of truth. SQLite is a derived, immutable runtime index:

```bash
uv run clingrounder terminology build \
  --source data/processed/full_concepts.jsonl \
  --cache-dir .cache/clingrounder/terminology

uv run clingrounder terminology inspect \
  --index .cache/clingrounder/terminology/<fingerprint>.sqlite3 \
  --query metformin \
  --entity-type DRUG \
  --code-system RxNorm

uv run clingrounder terminology query-set \
  --alias-overlay outputs/mining/knowledge/reviewed-aliases.jsonl \
  --output outputs/mining/benchmark/queries.jsonl \
  --manifest-output outputs/mining/benchmark/query-manifest.json
```

The cache key includes source SHA-256, schema version, and normalization version. Runtime opens
read-only, query-only, thread-local connections and never rebuilds a stale index implicitly.

## CLI And Validation

```text
clingrounder pipeline run
clingrounder terminology build|inspect|query-set|benchmark
clingrounder validate
clingrounder release audit
clingrounder evaluate
clingrounder-research model ...
clingrounder-research data ...
clingrounder-benchmark list
clingrounder-benchmark phase1 ...
```

`python -m clingrounder.cli` is the same operational entrypoint as `clingrounder`. Research and
benchmark commands must use their scoped entrypoints; the dispatcher no longer exposes one
all-purpose installed command.

Terminology mining deliberately has separate stages. `mining/crosswalk.py` emits exact, ambiguous,
lexical, and unmatched lookup evidence. `mining/crosswalk_links.py` can attach only policy-pinned,
exact-unique rows to source annotations without changing spans or overwriting existing concepts.
Alias promotion remains a later reviewed operation. Search these boundaries with:

```bash
rg "crosswalk_mentions|materialize_exact_crosswalk_links|propose_linked_aliases" \
  src/clingrounder/mining src/clingrounder/cli tests
```

`mining/cooccurrence.py` can aggregate neutral evidence within a sentence or a hash-validated
source block. Typed preferred code systems select one canonical endpoint when an annotation keeps
multiple provenance links. `mining/graph_knowledge.py` uses the same selector during graph
compilation and rejects selected codes absent from loaded canonical terminology. Search with:

```bash
rg "preferred_code_systems_by_entity_type|context_scope|canonical-concepts-only" \
  src/clingrounder/mining src/clingrounder/cli tests
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
rg "class .*Port" src/clingrounder/pipeline src/clingrounder/terminology
rg "PipelineConfig|from_mapping" src tests configs
rg "INVARIANT:|SCALING:|MODEL:|LICENSE:|PRIVACY:" src tests
rg "EntityAnnotation|RelationAnnotation" src/clingrounder/schema tests
rg "TerminologyRepository|exact_lookup|search" src tests
rg "EvaluationAdapter" src/clingrounder/evaluation src/clingrounder/benchmarks tests
rg "ValidationProfile" src tests
```

Comments with `INVARIANT:`, `SCALING:`, `MODEL:`, `LICENSE:`, and `PRIVACY:` explain non-obvious
safety, concurrency, projection, source-policy, and data-handling decisions. Ordinary control flow
should remain self-explanatory rather than narrated.
