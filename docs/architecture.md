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

## Pipeline

```text
Raw clinical text
  -> ClinicalDocument
  -> section and sentence splitting
  -> rule NER or model NER
  -> assertion classification
  -> candidate retrieval
  -> entity linking
  -> relation extraction
  -> KG validation
  -> ClinicalPrediction JSON
  -> evaluation and error analysis
```

## Retrieval

Candidate generation must avoid brute-force mention-to-dictionary comparison. The intended order is:

```text
mention
  -> exact lookup
  -> abbreviation lookup
  -> fuzzy top-k
  -> character n-gram top-k
  -> BM25 top-k
  -> optional dense top-k
  -> merge and deduplicate
  -> type/code-system filter
  -> rerank top-k only
```

Dense retrieval and cross-encoder reranking are extension points. The correctness gate is
candidate recall at 20 before expensive reranking is added.

## Data Storage

Prototype data stays in JSONL, Parquet-compatible tables, DuckDB/SQLite-ready graph tables, and
small local indexes. A graph database is deferred until interactive graph traversal becomes a real
requirement.

## Agentic Workflow

Agents should read `AGENTS.md`, then the smallest relevant docs and modules. Work should be split by
module boundaries such as schema, offset mapping, context rules, dictionary linking, or KG
constraints. Large cross-repo refactors are out of scope unless explicitly requested.
