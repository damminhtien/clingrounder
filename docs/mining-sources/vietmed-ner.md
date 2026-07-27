# VietMed-NER

## Source Identity And Permission

VietMed-NER is a Vietnamese spoken-medical NER corpus with 18 source entity types. Dataset bytes
are pinned to Hugging Face revision `e3d0393c733858402a7c04228f45d351d2ce6d8f`; the published
XLM-R model is pinned separately to revision `cccffb7de14423114f7d4bafc9f736b9d866e446`.

The data owner confirmed on 2026-07-27 that this project may use the dataset and model for training
and inference. The registry records that permission as
`research-use-confirmed-2026-07-27`. Source redistribution remains prohibited because the public
model card does not declare an SPDX license. Generated manifests and aggregate statistics may be
committed; Parquet, audio, transcripts and model weights stay in ignored artifact storage.

The three immutable Parquet inputs are:

| Source split | SHA-256 | Bytes |
| --- | --- | ---: |
| train | `5d04692de75257c5392ad7a1b4cc4fbdd078f4b454719dbae46b89d284452a3c` | 441,918,947 |
| validation | `bf1caebca2d49d6f2fb91bfb7297d32b083be87a31fb64cadf05b6eadcbd80ed` | 111,116,959 |
| test | `fd9c5e5778ded8d01512e1acf19c4e75a788c861d43299683e16d13a2b05228a` | 340,472,951 |

## Parsing And Offset Contract

`VietMedNerParquetParser` projects only `words`, `labels`, `text` and `duration`; it never loads the
embedded audio column. It aligns every source token monotonically against the untouched transcript
and rejects non-whitespace gaps. The BIO labeler then validates each entity envelope against that
same text.

Observed import:

| Measure | Value |
| --- | ---: |
| documents | 9,267 |
| source-human annotation proposals | 22,974 |
| offset mismatches | 0 |
| train documents | 4,616 |
| development documents | 1,154 |
| source test documents | 3,497 |

The official `train`, `validation` and `test` assignments are preserved. Only the source
`validation` name is normalized to the repository name `development`; no document is hashed or
randomly reassigned.

## Source Taxonomy

The import retains all 18 labels rather than pretending they are Phase 1 gold. Counts over all
splits are:

| Source label | Count | Phase 1 use |
| --- | ---: | --- |
| `DISEASESYMTOM` | 5,188 | disease/symptom verifier evidence; Qwen must decide the final type |
| `DRUGCHEMICAL` | 2,114 | drug verifier evidence |
| `DIAGNOSTICS` | 763 | lab-test verifier evidence |
| `ORGAN` | 3,717 | representation/context only |
| `UNITCALIBRATOR` | 1,299 | dose/lab-unit hard negatives and context |
| `AGE` | 1,187 | non-entity hard-negative context for Phase 1 |
| `DATETIME` | 1,531 | non-entity hard-negative context for Phase 1 |
| `TREATMENT` | 1,150 | treatment context; Phase 1 has no procedure label |
| all remaining ten labels | 4,025 | representation/context only |

`DISEASESYMTOM` is the spelling in the published source and is preserved for reproducibility.
The importer does not infer diagnosis versus symptom, assertion, medical code or relation.

## Derived Model Artifacts

The 18-type span export contains 9,267 raw-text records and all 22,974 annotations:

```text
outputs/mining/model-datasets/vietmed-ner-e3d0393c-18type/
```

Its span JSONL SHA-256 is
`54430c37228099d1c97d7e703ba51f3153319c23d055b39146cce61a0282f90b`.
This artifact supports source-task model evaluation or continued training while preserving source
labels.

The train-only exact-quote curriculum contains 4,616 records and 10,888 entity quotes:

```text
outputs/mining/model-datasets/vietmed-ner-qwen-exact-quote-v1/
```

Its JSONL SHA-256 is
`cfe2274d07621d63698c9d667c15ad6a509f202a6c988c92cbed011159dfa09f`.
No development/test record, target-task crosswalk or offset appears in the assistant target. It is
intended as a Vietnamese medical quote-extraction curriculum before Phase 1 specialization.

## Published Model As Verifier

The published `xlm-roberta-base-VietMed-NER` checkpoint has 277,481,509 parameters. Its repository
revision, subfolder and count are locked in
`configs/models/phase1-vietmed-ner-verifier-2026-07-27.yaml`.
`HuggingFaceSourceTokenClassifierAdapter` preserves the source taxonomy and projects fast-tokenizer
offsets back to raw text. The Phase 1 compatibility layer exposes only:

```text
DISEASESYMTOM -> {CHẨN_ĐOÁN, TRIỆU_CHỨNG}
DRUGCHEMICAL  -> {THUỐC}
DIAGNOSTICS   -> {TÊN_XÉT_NGHIỆM}
```

These are candidates for Qwen adjudication, not final predictions. The Qwen consensus function
requires at least one `qwen.*` source, so VietMed-NER plus a rule source still cannot emit an entity
without Qwen. The combined Qwen3-8B and VietMed-NER budget is 8,468,216,869 parameters; its separate
experiment config is `phase1-qwen3-8b-vietmed-verifier-2026-07-27.yaml`. The original Qwen config
remains an unchanged control.

## Promotion Boundary

- Allowed: source-task training, exact-quote curriculum, model inference and verifier evidence.
- Not direct runtime output: VietMed-NER model entities must be confirmed by Qwen or another
  calibrated target-task model.
- Not target gold: `DISEASESYMTOM` cannot decide disease versus symptom.
- Not terminology truth: source strings do not acquire ICD-10 or RxNorm codes.
- Not assertion supervision: the source does not establish Phase 1 negation/history/family labels.
- Not redistributable: Parquet, audio, transcripts and model weights remain outside Git artifacts.

This boundary specifically prevents the failed XLM-R bulk-union behavior from recurring. A
Vietnamese model may raise a candidate for adjudication, but it cannot add an entity to a
submission by itself.

## Reproduce

Acquire and parse the exact source release:

```bash
uv run medical-kg data run --plan configs/mining/vietmed_ner.yaml

uv run medical-kg data label propose \
  --documents outputs/mining/vietmed-ner-e3d0393c/documents.jsonl \
  --adapter medical_kg_nlp.mining.labelers.vietmed_ner:create_vietmed_ner_source_labeler \
  --adapter-config configs/mining/labelers/vietmed_ner.yaml \
  --output outputs/mining/vietmed-ner-e3d0393c/source_annotations.jsonl
```

Freeze the source partition and export the full source-task span dataset:

```bash
uv run medical-kg data dataset freeze-source-splits \
  --documents outputs/mining/vietmed-ner-e3d0393c/documents.jsonl \
  --map train=train --map validation=development --map test=test \
  --output outputs/mining/vietmed-ner-e3d0393c/source_split_manifest.json

uv run medical-kg data dataset export-spans \
  --documents outputs/mining/vietmed-ner-e3d0393c/documents.jsonl \
  --annotations outputs/mining/vietmed-ner-e3d0393c/source_annotations.jsonl \
  --split-manifest outputs/mining/vietmed-ner-e3d0393c/source_split_manifest.json \
  --output outputs/mining/model-datasets/vietmed-ner-e3d0393c-18type/spans.jsonl \
  --manifest-output outputs/mining/model-datasets/vietmed-ner-e3d0393c-18type/manifest.json \
  --max-characters 1200
```

Build the train-only Qwen curriculum by passing all 18 source labels to
`build-exact-quote-curriculum`. The expected build fingerprint is
`a1099af3ea44afee7fc32e74bbc60e982a1041afbcdac25baaa16665c63ba551`.

On a machine with the `ml` extra and a local pinned checkpoint, build support evidence:

```bash
hf download leduckhai/VietMed-NER \
  --revision cccffb7de14423114f7d4bafc9f736b9d866e446 \
  --include 'xlm-roberta-base-VietMed-NER/*'

uv run medical-kg benchmark phase1 qwen build-vietnamese-support \
  --config configs/models/phase1-vietmed-ner-verifier-2026-07-27.yaml \
  --documents outputs/mining/phase1-round2-hosted-2026-07-27/documents.jsonl \
  --source-archive-sha256 989d82404a9c1f3739e15d68a1e69d0f1f90d35c93c04ab0988e071fc1525545 \
  --output-dir outputs/models/phase1-vietmed-ner-round2-support
```

Then pass the generated `support/` directory to Qwen:

```bash
uv run medical-kg benchmark phase1 qwen propose \
  --config configs/models/phase1-qwen3-8b-vietmed-verifier-2026-07-27.yaml \
  --documents outputs/mining/phase1-round2-hosted-2026-07-27/documents.jsonl \
  --source-archive-sha256 989d82404a9c1f3739e15d68a1e69d0f1f90d35c93c04ab0988e071fc1525545 \
  --support-source vietmed.ner=outputs/models/phase1-vietmed-ner-round2-support/support \
  --output-dir outputs/models/phase1-qwen3-vietmed-round2
```
