# Dictionary Linker

Use this skill for ICD/RxNorm/UMLS-like dictionaries, alias tables, candidate generation, and
linking evaluation.

## Scope

- Load dictionary JSONL into typed concept entries.
- Maintain exact, abbreviation, fuzzy, BM25, and optional dense retrieval.
- Keep candidate generation dictionary-constrained.
- Evaluate candidate recall at k before reranker changes.

## Guardrails

- Never output a code absent from the dictionary.
- Never map drug entities to ICD-10 or disease entities to RxNorm.
- Avoid brute-force full-dictionary comparison in online candidate generation.
