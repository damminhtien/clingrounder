# ADR 0005: Scale Terminology Retrieval Without Expanding Recognition Blindly

## Status

Accepted.

## Context

The Phase 1 full configuration loads the complete processed RxNorm July 2026 release (73,912
concepts) for normalization. Recognition remains a smaller reviewed terminology set because making
every RxNorm string a NER trigger creates false-positive spans before retrieval can correct them.
Recognition coverage and normalization coverage are therefore separate controls.

An all-source diagnostic enabled exact, abbreviation, fuzzy, character n-gram, and BM25 retrieval
over the full terminology store. On 100 notes it processed 0.20 documents/second, compared with
12.68 documents/second for the optimized exact path. Candidate generation became the dominant
stage, while reviewed manual-gold score decreased from 52.4989 to 52.4261. More candidate recall is
not useful while candidate precision and abstention are the limiting factors.

## Decision

- Keep the full RxNorm release in the normalization store; do not fall back to seed-only linking.
- Build only retrieval indexes selected by configuration.
- Load terminology entries first, merge them, and construct each retained lookup index once.
- Keep exact retrieval as the production default until another source improves both runtime-gated
  accuracy and candidate score.
- Keep the all-source configuration as an explicit diagnostic, not a submission default.
- Do not add Chroma without a versioned embedding model and a dense-retrieval benchmark. Chroma
  collections are embedding-oriented, so adding it now would introduce a second retrieval regime
  without evidence that dense recall is the current bottleneck.
- Do not require Elasticsearch in the offline core path. Its BM25/vector and kNN features are
  useful at service scale, but operating an external server is disproportionate for the current
  74k-concept store and conflicts with the offline submission path.
- SQLite FTS5 with a trigram tokenizer is the preferred next persistent lexical experiment. It is
  embedded, supports BM25 and substring indexing, and can be rebuilt atomically from terminology
  fingerprints. It must still pass the same accuracy gate before promotion.
- Do not add ANN solely to reduce latency. Faiss-style ANN trades exactness for speed and index
  build cost; current candidate recall is already high while candidate precision is weak.

## Consequences

Terminology releases and reviewed aliases may change without changing code. Run manifests retain
source fingerprints, runtime telemetry records initialization and processing separately, and the
runtime benchmark compares stage totals and output ZIP hashes. A persistent index may be added
behind the existing retriever interface later without changing pipeline or export contracts.

References:

- [SQLite FTS5](https://www.sqlite.org/fts5.html)
- [Chroma collection management](https://docs.trychroma.com/docs/collections/manage-collections)
- [Elasticsearch vector search](https://www.elastic.co/docs/solutions/search/vector)
- [Faiss index selection](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
