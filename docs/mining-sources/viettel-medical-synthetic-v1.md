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
uv run medical-kg benchmark phase1 model-data augment-user-synthetic \
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

## Limitations

- Synthetic `dev` and `test` are never calibration or challenge data.
- Template frequency is not evidence of real clinical prevalence.
- Source candidate mappings may encode source-specific annotation noise.
- The dataset must remain at or below 40% of a training batch.
- A model trained with this view must still be selected on human development and evaluated on a
  source-held-out human set.
