# CodiEsp

## Source And Acquisition

CodiEsp v1.4 is pinned to Zenodo record `3837305`, archive SHA-256
`52b290233906a2eb589ac7b1d9429adeac88f6a9cae8b0d3c180504afbe61688`, under CC BY 4.0. The parser
reads Spanish source text only and excludes machine-translated English files.

## Parsing And Curation

| Measure | Value |
| --- | ---: |
| documents | 3,751 |
| source-human annotations | 18,435 |
| diagnosis/finding links | 14,305 ICD-10-CM |
| procedure links | 4,130 ICD-10-PCS |
| discontinuous annotations | 3,707 |
| import rows needing review | 11 |
| exact contiguous training spans | 14,726 |

Discontinuous source geometry is preserved. The token-classifier view rejects discontinuous spans,
spans longer than 256 characters, and import issues; it does not train on the raw envelope.

## Extracted Terminology And Graph Knowledge

Only official CodiEsp train documents may promote retrieval aliases for leakage-safe evaluation.
The train overlay has 393 aliases. On source-held-out data it changes exact hit@1 and FTS recall@20:

| Split | Base hit@1 | Enriched hit@1 | Base recall@20 | Enriched recall@20 |
| --- | ---: | ---: | ---: | ---: |
| dev | 0.35% | 18.09% | 8.32% | 28.55% |
| test | 0.36% | 19.32% | 8.13% | 30.65% |

Unknown ICD-10-CM/PCS codes are never admitted to TT06 by code shape. There are 400 impossible dev
targets and 376 impossible test targets absent from TT06, and they remain explicit failures.

The train-only relation pass emits literal symmetric `CO_OCCURS_WITH` evidence. It produced 225
semantic pairs and 721 occurrences; after requiring canonical disease endpoints, 180 graph edges
with 604 evidence occurrences remain. Co-occurrence is never promoted to `TREATS`, `CAUSES`, or
`HAS_SYMPTOM`.

## Promotion Boundary

The train-only terminology overlay and graph evidence are runtime opt-in. Source dev/test labels are
query-only. CodiEsp supports Spanish NER/linking and weak graph reranking evidence; it does not prove
Vietnamese aliases, assertion status, or causal clinical relations.

Raw and curated artifacts are under `outputs/mining/codiesp-zenodo-3837305/`; leakage-safe knowledge
is under `outputs/mining/knowledge/codiesp-icd10-split-2026-07-18/`.
