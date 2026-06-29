# 0003: Hybrid Retrieval

## Status

Accepted.

## Context

Medical entity linking needs exact lexical matching for codes, disease names, drugs, abbreviations,
and synonyms. Dense retrieval can help with paraphrases but should not replace dictionary safety.

## Decision

Use hybrid candidate generation:

```text
exact + abbreviation + fuzzy + BM25 + optional dense retrieval
```

Merge and filter candidates before reranking. Optimize for candidate recall at 20, then add learned
reranking only over the small candidate set.

## Consequences

- Avoids brute-force mention-to-dictionary comparison.
- Keeps outputs constrained to known dictionary concepts.
- Allows dense retrieval and FAISS/Qdrant to be added without changing the linker contract.
