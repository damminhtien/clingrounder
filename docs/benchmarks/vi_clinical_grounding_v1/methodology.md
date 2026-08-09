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
