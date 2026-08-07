# Viettel Medical Synthetic v1

## Role

`viettel_medical_synthetic_v1.zip` is a user-provided synthetic Phase 1 corpus. It contains 800
generated Vietnamese records:

| Split in source archive | Records | Runtime policy |
| --- | ---: | --- |
| `train` | 600 | Eligible for bounded NER training |
| `dev` | 100 | Validated, never exported |
| `test` | 100 | Validated, never exported |

The archive is not clinician-reviewed gold. Its templates imitate BTC-shaped clinical,
question-answer, terse EHR, mixed, and noisy text. Diagnosis and drug code surfaces were derived
from consistent mappings in user-provided reference ground truth. Consequently, the dataset is
useful for quote extraction and document-style robustness, but not as an independent candidate
evaluation set.

Pinned source SHA-256:

```text
ad5baa1d4c4ef41265124518ca59def4744511f96c9a254a3b051c43a757500b
```

## Audit

The importer reads JSONL directly from the ZIP and does not execute the bundled generator or
validator scripts. It validates all 800 records before selecting training rows.

Observed source totals:

- 17,850 entities;
- 5,096 symptoms;
- 4,069 lab tests;
- 3,863 lab results;
- 2,813 diagnoses;
- 2,009 medications;
- zero offset, duplicate-identity, overlap, or cross-split normalized-document errors.

Against the authorized Round 2 input:

- zero exact or normalized document matches;
- maximum per-document 12-token-shingle overlap `0.0078125`;
- no synthetic document reaches `0.05` overlap.

Against the scored 31.2236 proposal artifact, the archive covers about 15.0% of entity occurrences
but only 4.8% of unique type-plus-surface pairs. This confirms that it is a curriculum supplement,
not a substitute for model review on Round 2.

## Training View

The deterministic build uses the 81 human train chunks and leaves all 20 human development chunks
unchanged. It selects 54 synthetic train records, nine from each supplied style, producing exactly
a 40% synthetic share of training records.

Candidate and assertion fields are audited but removed from the NER span view. Only raw text,
five-type labels, and exact source offsets enter supervision.

```bash
uv run clingrounder-benchmark phase1 model-data augment-user-synthetic \
  --archive /path/to/viettel_medical_synthetic_v1.zip \
  --archive-sha256 \
    ad5baa1d4c4ef41265124518ca59def4744511f96c9a254a3b051c43a757500b \
  --source-dataset \
    outputs/mining/model-datasets/phase1-manual-five-type-v1/spans.jsonl \
  --source-manifest \
    outputs/mining/model-datasets/phase1-manual-five-type-v1/manifest.json \
  --output-dir \
    outputs/mining/model-datasets/phase1-manual-user-synthetic-v1 \
  --max-synthetic-fraction 0.4
```

Expected output identity:

```text
build_key: 065bbaa6d0582d2baceb2a39af3737511452013c8197db9e87c246d71abc32b4
spans.jsonl: eccf5be6e0e98e4b07e6a1508cf3ab23e3243126567608f17760eca77ca03c99
records: 155
entities: 3215
```

## Qwen Missing-Entity Curriculum

The public 31.2236 artifact showed that iterative, missing-only review has materially higher recall
than a single extraction pass. The training view therefore derives deterministic reviewer
examples from train records: a stable mask exposes part of the trusted entity set as
`EXISTING_ENTITIES`, while the assistant target contains only the held-out entities.

Development records are never converted into reviewer training examples. The reviewer prompt does
not expose trusted offsets; at runtime, quoted text is projected back to every still-unlabeled
occurrence in the immutable source. This avoids the first-occurrence error caused by
`str.index()`-style projection.

```bash
uv run clingrounder-benchmark phase1 model-data build-qwen \
  --source-dataset \
    outputs/mining/model-datasets/phase1-manual-user-synthetic-v1/spans.jsonl \
  --source-manifest \
    outputs/mining/model-datasets/phase1-manual-user-synthetic-v1/manifest.json \
  --output-dir \
    outputs/mining/model-datasets/phase1-qwen-user-synthetic-review-v1 \
  --review-masks-per-train-record 2 \
  --review-keep-fraction 0.5 \
  --review-seed phase1-qwen-review-missing-v1
```

Expected identity:

```text
build_fingerprint: 46d451f7df5af38521460f1130fea653cf26ef95d247058d844dffbe896be6e9
extraction.jsonl: ade75d4b7168a0c4595cb34af52af57e1db7d1dfc99017754b1ad7a38fb6f4b0
review_missing.jsonl: 1da0405325e3bce29799575aa218a03872929fd5441959535a8c2717daa1e467
extraction records: 155
review records: 270
```

## Limitations

- Synthetic `dev` and `test` are never calibration or challenge data.
- Template frequency is not evidence of real clinical prevalence.
- Source candidate mappings may encode source-specific annotation noise.
- The dataset must remain at or below 40% of a training batch.
- A model trained with this view must still be selected on human development and evaluated on a
  source-held-out human set.
