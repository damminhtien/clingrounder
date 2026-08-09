# Benchmark methodology

`vi-clinical-grounding-v1` is a product benchmark contract, not a competition scorer. It reports
exact raw-span/type matching, assertion status, terminology linking, relations, and runtime in
separate sections.

## Pilot limitations

The checked-in `0.1.0` fixture is synthetic and small. It is useful for smoke tests, regression
tests, and reproducibility checks only. It is not evidence of clinical generalization. A public
clinical release requires human review, license provenance, template grouping, and a held-out test
set that is not used to tune rules.

## Output contract

Each run records the profile and input fingerprints, Git commit, exact predictions, per-document
errors, assertion confusion counts, runtime percentiles, and a Markdown summary. The `suite`
command runs named profiles independently and adds a deterministic ablation index; it does not
merge predictions or hide profile provenance. Runtime numbers are machine-dependent and must be
compared with repeated runs and tolerances.

## Promotion policy

The primary metric is exact entity micro-F1. A change must improve it while preserving candidate
recall, positive assertion macro-F1, offset validity, deterministic ordering, and bounded p95
latency. Machine-sensitive runtime values are reported with tolerances, not treated as absolute
scientific claims.

## Publication audit

Run the dataset audit before treating a result as public clinical evidence:

```bash
clingrounder-benchmark audit \
  --benchmark benchmarks/vi_clinical_grounding_v1 \
  --output artifacts/benchmarks/vi-clinical-grounding-v1/audit.json \
  --strict
```

The audit verifies declared split counts and SHA-256 fingerprints, unique document IDs,
normalized-text and template leakage, license metadata, test isolation, and human-review coverage.
For a human-reviewed release, ``dataset_manifest.yaml`` must also point to a hashed agreement
report:

```yaml
review:
  reviewers_required: 2
  double_review_fraction: 0.10
  agreement_targets:
    span_type: 0.90
    assertion: 0.85
    relation: 0.80
  agreement_report: review/agreement.json
  agreement_report_sha256: "<sha256>"
```

The report uses the neutral ``clingrounder.review-agreement.v1`` schema and records measured
span/type, assertion, and relation agreement plus the double-review fraction. The audit checks
the report fingerprint, dataset identity, and every declared target; a ``human_reviewed`` boolean
without this evidence cannot pass the clinical-evidence gate. Reports contain no raw note text or
reviewer identity by default. A synthetic or review-pending dataset can pass the engineering
checks while correctly failing the clinical-evidence gate.

Generate the artifact from the existing task-neutral review quality output instead of editing JSON
by hand:

```bash
clingrounder-research data review quality \
  --documents outputs/gold/documents.jsonl \
  --proposals outputs/gold/annotations.jsonl \
  --output outputs/review-quality.json \
  --dataset-id vi-clinical-grounding-v1 \
  --dataset-version 1.0.0 \
  --benchmark-output outputs/review-agreement.json
```

## Review handoff

The repository provides `clingrounder-benchmark review-pack` for independent annotation. It derives
review assignments from the dataset and a seed, writes a coordinator-only source-ID map, and emits
gold-blind reviewer JSONL. The reviewer payload intentionally contains no `entities` or `relations`
from the benchmark input. This prevents checked-in synthetic labels, or labels from a later licensed
snapshot, from silently becoming reviewer supervision.

The coordinator must retain the generated `manifest.json`, source fingerprints, and
`coordinator_document_map.jsonl`. After two reviewers complete their annotations, validate and join
the edited files with:

```bash
clingrounder-benchmark review-pack-import \
  --benchmark benchmarks/vi_clinical_grounding_v1 \
  --pack artifacts/review-packs/vi-clinical-grounding-v1 \
  --split test \
  --output artifacts/review-imports/vi-clinical-grounding-v1
```

The resulting `adjudication.jsonl` preserves each independent submission and marks exact
agreement versus `needs_adjudication`. It is deliberately not a gold dataset. Resolve every
adjudication item, export a gold layer, validate its agreement artifact, and only then update the
dataset manifest. The pilot remains ineligible for clinical claims until that evidence is
genuinely produced.
