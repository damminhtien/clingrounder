# VietBioNER

## Source Identity And Acquisition

VietBioNER is pinned to Git commit `19ba70a5947d1be72906d407c860b1666b9337e9` under CC BY 4.0.
The immutable ZIP SHA-256 is
`4a863c9659bdf97754d5f335133e9a0f6f748d69be4902c3af40f25d6f49738a`.
`configs/mining/vietbioner.yaml` downloads that exact archive into the content-addressed artifact
store. The source is Vietnamese biomedical literature, mainly tuberculosis and HIV research; it is
not a clinical-note corpus.

The BRAT parser preserves the source text used by each annotator. It emitted 70 annotator documents
from one source artifact, with 63 unique texts. Fourteen documents form seven exact-text duplicate
groups because the same source text was independently annotated twice. No document is translated,
normalized or rewritten before offsets are checked.

## Parsing And Offset Contract

`medical_kg_nlp.mining.labelers.brat` reads the source `.ann` files and validates every raw
`[start,end)` envelope against the corresponding document. Original BRAT segments and annotated
text remain in metadata. The import produced:

| Measure | Value |
| --- | ---: |
| annotator documents | 70 |
| unique source texts | 63 |
| source-human annotations | 3,574 |
| exact duplicate groups | 7 |
| discontinuous source annotations | 248 |
| offset mismatches | 0 |
| mean document length | 3,373 characters |
| maximum document length | 8,798 characters |

The source frequently marks a phrase split across a PDF line break as discontinuous. After
reconciliation, 208 records retain multiple BRAT segments. Of the 128 records relevant to the NER
view, all gaps contain whitespace/newlines only; no medical token is skipped. Their raw envelope is
therefore a valid contiguous token-classification span. The segment list is still retained because
this conclusion is source-specific and must not become a generic assumption for BRAT corpora.

OCR and extraction noise is substantial: examples include `lao phối`, `lao phôi`, malformed
abbreviations and words split by newlines. This is useful robustness supervision for a model, but it
is unsafe as an automatic canonical alias source.

## Source Labels And Reconciliation

The source labels and internal import types are:

| Source label | Source count | Internal type | Allowed use |
| --- | ---: | --- | --- |
| `Symptom_and_Disease` | 2,119 | `FINDING` | broad NER supervision only |
| `DiagnosticProcedure` | 549 | `PROCEDURE` | procedure NER supervision |
| `DateTime` | 306 | `OTHER` | retained provenance, excluded from medical NER view |
| `Location` | 375 | `OTHER` | retained provenance, excluded from medical NER view |
| `Organisation` | 225 | `OTHER` | retained provenance, excluded from medical NER view |

`FINDING` deliberately does not distinguish diagnosis from symptom. The importer does not infer a
narrower clinical type, assertion, relation or medical code.

Exact duplicate reconciliation keeps labels present at the same type and span in both annotator
copies. Non-duplicate documents keep their source-human labels; disagreements from duplicate pairs
go to review instead of being unioned. The reconciled output has 63 documents, 3,109 silver
annotations and 164 review hypotheses. Agreement is not high enough to call the result gold:

| Label | Exact Jaccard |
| --- | ---: |
| `Symptom_and_Disease` | 0.6444 |
| `DiagnosticProcedure` | 0.5328 |
| `DateTime` | 0.6154 |
| `Location` | 0.8718 |
| `Organisation` | 0.9200 |
| all labels, micro | 0.6466 |
| all duplicate pairs, macro | 0.6570 |

The materialized v4 snapshot is
`vietbioner-19ba70a-reconciled-silver-v4-736f876c7492df01`. It contains Parquet document and
annotation shards, 51 train documents and 12 development documents. Its manifest SHA-256 is
`41cbe571649aba2717c6337cb02f3aac53794509a5b212635c4b72c8fb2bb41e`; source fingerprints pin the
archive, agreement report and document map. V1/V2 were historical manifest-only snapshots and must
not be used for a new model run.

## Model NER View

`configs/mining/curation/vietbioner-ner.yaml` creates a source-specific view. It accepts only silver
`FINDING` and `PROCEDURE` records, preserves the broad source semantics and rejects all 802 `OTHER`
labels. The accepted 2,307 spans contain 1,880 findings and 427 procedures.

The model-neutral export has 217 raw-text chunks at a 1,200-character soft limit: 180 train chunks,
37 development chunks and three empty negative chunks. It preserves all 2,307 accepted spans. Span
dataset SHA-256 is `76d5eb3ee0eae984864f0769388e61d1a5ec38ea9c5a264121c4d13d2ab2608a`.

This view is suitable for pretraining or domain adaptation of Vietnamese biomedical NER. It is not
an evaluation of clinical-note NER, and it provides no supervision for assertions, candidate codes
or relations.

## Terminology And Recognition Experiments

The all-source mention inventory contains 768 normalized entries. The pinned TT06/RxNorm exact
crosswalk resolves only eight entries covering 22 of 3,109 occurrences. Another 248 entries are
unmatched and 512 are skipped because their source labels do not permit a terminology query. Every
hit remains `review_required`; an exact string cannot resolve the source's diagnosis-versus-symptom
ambiguity.

The v4 recognition experiment is leakage-safe: only the 51 train documents build the inventory.
That inventory has 623 entries and five type conflicts. The pinned promotion policy accepted 87
code-free terms, 51 `FINDING` and 36 `PROCEDURE`, while rejecting 536 rows.

On the 12 development documents, the compact exact matcher produced:

| Metric | All | `FINDING` | `PROCEDURE` |
| --- | ---: | ---: | ---: |
| precision | 0.5402 | 0.6481 | 0.1216 |
| recall | 0.5052 | 0.6392 | 0.0947 |
| F1 | 0.5221 | 0.6436 | 0.1065 |
| true positives | 195 | 186 | 9 |
| false negatives | 191 | 105 | 86 |
| false positives | 166 | 101 | 65 |

Of 386 development occurrences, 252 have a normalized type/mention pair observed in train and 134
belong to 92 unseen pairs. The remaining errors also expose the limitation of dictionary scanning:
64 false negatives and 78 false positives overlap the correct boundary. Common procedure fragments
such as `AFB`, `nhuộm soi`, `cấy` and `MGIT` are especially unsafe without context. The v4 F1 is
below the historical split's 0.612, so neither recognition artifact is promoted to runtime.

The benchmark's baseline dictionary contains no matching `FINDING`/`PROCEDURE` concepts; its zero
score is expected. The reported enriched score measures the VietBioNER train-only dictionary, not
an end-to-end pipeline improvement.

## Promotion Boundary

- Training only: reconciled raw spans and the filtered NER model view.
- Review/diagnostic only: mention inventory, TT06 exact crosswalk, conflicts and recognition errors.
- Not runtime: both v1 and v4 recognition dictionaries because of boundary false positives and
  poor procedure precision.
- Never inferred: diagnosis versus symptom, patient assertion, relations or medical codes.
- Never canonical aliases: OCR variants and source abbreviations without independent terminology
  evidence and human review.

Primary artifacts:

```text
outputs/mining/vietbioner-19ba70a/
outputs/mining/snapshots/vietbioner-19ba70a-reconciled-silver-v4/
outputs/mining/model_datasets/vietbioner-19ba70a-reconciled-silver-v4/
outputs/mining/knowledge/vietbioner-recognition-v4/
```

## Reproduce

Acquire, import and reconcile the source:

```bash
uv run medical-kg data run --plan configs/mining/vietbioner.yaml

uv run medical-kg data label propose \
  --documents outputs/mining/vietbioner-19ba70a/documents.jsonl \
  --adapter medical_kg_nlp.mining.labelers.brat:create_brat_archive_labeler \
  --adapter-config configs/mining/labelers/vietbioner.yaml \
  --output outputs/mining/vietbioner-19ba70a/source_annotations.jsonl

uv run medical-kg data dataset reconcile-duplicates \
  --documents outputs/mining/vietbioner-19ba70a/documents.jsonl \
  --annotations outputs/mining/vietbioner-19ba70a/source_annotations.jsonl \
  --documents-output outputs/mining/vietbioner-19ba70a/reconciled/documents.jsonl \
  --annotations-output outputs/mining/vietbioner-19ba70a/reconciled/training_annotations.jsonl \
  --review-output outputs/mining/vietbioner-19ba70a/reconciled/review_annotations.jsonl \
  --mapping-output outputs/mining/vietbioner-19ba70a/reconciled/document_map.jsonl \
  --report-output outputs/mining/vietbioner-19ba70a/reconciled/agreement_report.json \
  --labeler-id vietbioner-exact-duplicate-consensus:v1
```

Freeze the materialized split and create the model view:

```bash
uv run medical-kg data snapshot freeze \
  --documents outputs/mining/vietbioner-19ba70a/reconciled/documents.jsonl \
  --annotations outputs/mining/vietbioner-19ba70a/reconciled/training_annotations.jsonl \
  --artifacts outputs/mining/vietbioner-19ba70a/artifacts.jsonl \
  --source-fingerprint 3dbc1f703f3b7d8ca080ad9bfb324596b10a1cf7ac82456c04bf594062a1f01d \
  --source-fingerprint fe22b5358e7b974a88a6c75bcc3dc8f876255e3bbc45aff55d9765a59295ab2d \
  --version vietbioner-19ba70a-reconciled-silver-v4 \
  --created-at 2026-07-20T04:49:09Z \
  --development-fraction 0.3 \
  --output-dir outputs/mining/snapshots/vietbioner-19ba70a-reconciled-silver-v4 \
  --skip-agreement-gate

uv run medical-kg data dataset curate-annotations \
  --annotations outputs/mining/vietbioner-19ba70a/reconciled/training_annotations.jsonl \
  --policy configs/mining/curation/vietbioner-ner.yaml \
  --accepted-output outputs/mining/vietbioner-19ba70a/reconciled/model_ner_annotations.jsonl \
  --rejected-output outputs/mining/vietbioner-19ba70a/reconciled/model_ner_rejected_annotations.jsonl \
  --report-output outputs/mining/vietbioner-19ba70a/reconciled/model_ner_curation_report.json

uv run medical-kg data dataset export-spans \
  --documents outputs/mining/vietbioner-19ba70a/reconciled/documents.jsonl \
  --annotations outputs/mining/vietbioner-19ba70a/reconciled/model_ner_annotations.jsonl \
  --split-manifest outputs/mining/snapshots/vietbioner-19ba70a-reconciled-silver-v4/manifest.json \
  --output outputs/mining/model_datasets/vietbioner-19ba70a-reconciled-silver-v4/spans.jsonl \
  --manifest-output outputs/mining/model_datasets/vietbioner-19ba70a-reconciled-silver-v4/manifest.json \
  --entity-type FINDING --entity-type PROCEDURE --max-characters 1200
```

Build and benchmark recognition knowledge without reading development labels during compilation:

```bash
uv run medical-kg data lexicon build \
  --documents outputs/mining/vietbioner-19ba70a/reconciled/documents.jsonl \
  --annotations outputs/mining/vietbioner-19ba70a/reconciled/training_annotations.jsonl \
  --split-manifest outputs/mining/snapshots/vietbioner-19ba70a-reconciled-silver-v4/manifest.json \
  --split train \
  --output outputs/mining/knowledge/vietbioner-recognition-v4/train_inventory.jsonl \
  --conflicts-output outputs/mining/knowledge/vietbioner-recognition-v4/train_conflicts.jsonl \
  --report-output outputs/mining/knowledge/vietbioner-recognition-v4/train_inventory_report.json

uv run medical-kg data knowledge compile-recognition \
  --inventory outputs/mining/knowledge/vietbioner-recognition-v4/train_inventory.jsonl \
  --policy configs/mining/knowledge/vietbioner-recognition-v4.yaml \
  --baseline-dictionary data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl \
  --output outputs/mining/knowledge/vietbioner-recognition-v4/recognition_concepts.jsonl \
  --decisions-output outputs/mining/knowledge/vietbioner-recognition-v4/decisions.jsonl \
  --report-output outputs/mining/knowledge/vietbioner-recognition-v4/compilation_report.json

uv run medical-kg data knowledge benchmark-recognition \
  --documents outputs/mining/vietbioner-19ba70a/reconciled/documents.jsonl \
  --annotations outputs/mining/vietbioner-19ba70a/reconciled/training_annotations.jsonl \
  --baseline-dictionary data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl \
  --additional-dictionary outputs/mining/knowledge/vietbioner-recognition-v4/recognition_concepts.jsonl \
  --entity-type FINDING --entity-type PROCEDURE \
  --split-manifest outputs/mining/snapshots/vietbioner-19ba70a-reconciled-silver-v4/manifest.json \
  --split development \
  --output outputs/mining/knowledge/vietbioner-recognition-v4/development_recognition_benchmark.json
```
