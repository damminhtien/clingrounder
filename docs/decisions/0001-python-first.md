# 0001: Python-First Core

## Status

Accepted.

## Context

The project needs clinical NLP, dictionary linking, retrieval experiments, error analysis, and
future transformer integration. Python has the strongest local ecosystem for this phase.

## Decision

Use Python 3.11+ as the core runtime. Keep Rust or C++ for profiled bottlenecks only.

## Consequences

- Faster iteration for schema, retrieval, and evaluation work.
- Easier integration with PyTorch, Hugging Face, scikit-learn, DuckDB, Polars, and FAISS/Qdrant
  later.
- Performance-critical extensions must keep Python-callable interfaces and tests.
