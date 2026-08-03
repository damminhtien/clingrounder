# Contributor Workflow

This guide is the shortest path from a clean clone to a reviewable change.

## Setup

```bash
uv sync --extra dev
uv run pre-commit install
uv run medical-kg pipeline run \
  --config configs/pipeline/clinical-baseline.yaml \
  --input data/samples/sample_notes.jsonl \
  --output outputs/sample-predictions.jsonl
```

Without `uv`, create a virtual environment and install `.[dev]`.

## Before Editing

1. Read `AGENTS.md`.
2. Read [invariants.md](invariants.md) and [schema.md](schema.md).
3. Use [code-map.md](code-map.md) to identify the owning package and public port.
4. Search for the contract and nearby tests with `rg`.
5. Keep benchmark policy below `medical_kg_nlp.benchmarks`.

Useful searches:

```bash
rg "class .*Port" src/medical_kg_nlp/pipeline src/medical_kg_nlp/terminology
rg "EntityAnnotation|RelationAnnotation" src/medical_kg_nlp/schema tests
rg "INVARIANT:|SCALING:|MODEL:|LICENSE:|PRIVACY:" src tests
```

## Change Loop

1. Write or update the smallest focused test.
2. Implement behind the existing module boundary.
3. Run the targeted test file.
4. Run Ruff and mypy for the changed package.
5. Run the default suite before committing.
6. Update docs when a command, config, public contract, or behavior changes.

```bash
uv run pytest tests/test_context_rules.py -q
uv run ruff check src/medical_kg_nlp/context tests/test_context_rules.py
uv run mypy src
uv run pytest tests
```

## Extension Patterns

### Add an entity extractor

Implement `EntityExtractorPort`, preserve raw spans, and inject the adapter through
`PipelineComponents` or `PipelineFactory`. Model token offsets must be projected back to the source
string before creating an entity.

### Add a retriever

Implement `CandidateRetrieverPort`. Return terminology-backed candidates with source evidence;
leave qualification and final assignment to linking. Filter entity type and code system before the
result can be emitted.

### Add a terminology backend

Implement `TerminologyRepository`. Canonical source data remains immutable JSONL; derived indexes
must fingerprint source, schema, aliases, and normalization behavior.

### Add a data source

Create a `SourceConnectorPort` adapter, register access/license/retention policy, add an offline
fixture test, and write a processing dossier under `docs/mining-sources/`.

### Add a benchmark

Implement the benchmark plugin contract, keep its schema/configs/tests under task-owned
directories, and translate records through `EvaluationAdapter`. Core packages must not import it.

## Comments And Names

Public modules need a module docstring and explicit `__all__`. Names follow `*Port`, `*Adapter`,
`*Repository`, `*Factory`, and `*Report` ownership.

Use comments for non-obvious decisions:

- `INVARIANT:` offset, schema, type, or code safety;
- `SCALING:` caching, batching, concurrency, or storage decisions;
- `MODEL:` token projection, revision, or confidence provenance;
- `LICENSE:` source-use or redistribution constraints;
- `PRIVACY:` protected-data boundaries.

Do not narrate straightforward control flow.

## Test Tiers

```bash
# Fast unit and contract suite
uv run pytest tests

# Entire redistributable suite
uv run pytest -o addopts='' tests

# Optional tiers
uv run pytest -o addopts='' -m integration tests
uv run pytest -o addopts='' -m release tests
uv run pytest -o addopts='' -m benchmark tests
```

`private` tests require authorized local data. `model` tests require pinned local weights and may
not download from the network.

Schema, duplicate IDs, raw offset/text identity, entity/code-system compatibility, and relation
endpoint/evidence validity are hard gates in every tier.

## Public Release Check

```bash
uv run medical-kg release audit \
  --policy configs/repository/public-release.yaml \
  --root .
```

The audit inspects tracked bytes and credential patterns. It does not delete local data. See
[public-release.md](public-release.md).
