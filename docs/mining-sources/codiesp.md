# CodiEsp

## Source And Acquisition

CodiEsp v1.4 is pinned to Zenodo record `3837305`, archive SHA-256
`52b290233906a2eb589ac7b1d9429adeac88f6a9cae8b0d3c180504afbe61688`, under CC BY 4.0. The parser
reads Spanish source text only and excludes machine-translated English files.

Acquisition is declared in `configs/mining/codiesp.yaml`; the connector verifies both the SHA-256
and Zenodo MD5 before parsing. The archive remains one content-addressed source artifact, while
split and annotation provenance are retained per document.

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
| documents with source annotations | 1,000 |
| offset mismatches | 0 |

Discontinuous source geometry is preserved. The token-classifier view rejects discontinuous spans,
spans longer than 256 characters, and import issues; it does not train on the raw envelope.
The remaining unannotated source documents are not treated as negative-only clinical notes unless a
model-specific view explicitly opts into them.

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

### Graph Reranker Benchmark

The train-only graph was tested as a bounded score bonus over exact/BM25 terminology candidates.
Calibration used dev and evaluation used test; no dev/test relation evidence entered the graph.

| Context mode | Test graph-feature queries | Accuracy@1 before | Accuracy@1 after | MRR delta |
| --- | ---: | ---: | ---: | ---: |
| oracle linked context upper bound | 358 / 2,052 | 0.6730 | 0.6774 | +0.00194 |
| exact-unique predicted context | 333 / 2,052 | 0.6730 | 0.6769 | +0.00170 |

The predicted-context run used 1,614 anchors at 99.13% precision. It changed top-1 for 42 queries:
14 improved and 10 worsened. Recall@5/10/20 did not change because the graph only reorders existing
candidates. This is a small, positive reranking feature, not evidence that co-occurrence captures a
clinical causal relation. Full NER-predicted context remains a separate model benchmark; the result
above still uses gold target mentions.

## Promotion Boundary

The train-only terminology overlay and graph evidence are runtime opt-in. Source dev/test labels are
query-only. CodiEsp supports Spanish NER/linking and weak graph reranking evidence; it does not prove
Vietnamese aliases, assertion status, or causal clinical relations.

Raw and curated artifacts are under `outputs/mining/codiesp-zenodo-3837305/`; leakage-safe knowledge
is under `outputs/mining/knowledge/codiesp-icd10-split-2026-07-18/` and
`outputs/mining/knowledge/codiesp-train-cooccurrence-2026-07-19/`.

## Reproduce

```bash
uv run medical-kg-research data run --plan configs/mining/codiesp.yaml

uv run medical-kg-research data dataset curate-annotations \
  --annotations outputs/mining/codiesp-zenodo-3837305/source_annotations.jsonl \
  --policy configs/mining/curation/codiesp-contiguous-ner.yaml \
  --accepted-output outputs/mining/codiesp-zenodo-3837305/contiguous_training_annotations.jsonl \
  --rejected-output outputs/mining/codiesp-zenodo-3837305/noncontiguous_review_annotations.jsonl \
  --report-output outputs/mining/codiesp-zenodo-3837305/curation_report.json

uv run medical-kg-research data relation mine-cooccurrence \
  --documents outputs/mining/codiesp-zenodo-3837305/documents.jsonl \
  --annotations outputs/mining/codiesp-zenodo-3837305/source_annotations.jsonl \
  --policy configs/mining/relations/codiesp-train-cooccurrence.yaml \
  --split-manifest outputs/mining/snapshots/codiesp-zenodo-3837305-contiguous-silver-v1/manifest.json \
  --split train \
  --output outputs/mining/knowledge/codiesp-train-cooccurrence-2026-07-19/relations.jsonl \
  --report-output outputs/mining/knowledge/codiesp-train-cooccurrence-2026-07-19/cooccurrence_report.json
```
