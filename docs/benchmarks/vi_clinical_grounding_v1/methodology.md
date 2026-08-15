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

The single-run summary and suite index retain the benchmark-manifest SHA-256 and evaluated split
SHA-256. Suite rows also retain the profile source hash, resolved configuration fingerprint, and
terminology fingerprint. Publication references pin the portable profile source hash and
terminology fingerprint. The resolved fingerprint remains diagnostic because rebased absolute
resource paths intentionally make it host-specific. `verify-reference` therefore fails if the
same metric values were produced from a different dataset, profile source, or terminology release.
Older references without provenance fields remain metric-only evidence and should not be used for
new published measurements.

Every run also performs a post-inference validation pass against the active terminology
repository. The summary exposes these fail-closed gates separately from task quality metrics:

- `offset_validity`: `1.0` only when every returned entity span matches the source text and every
  document produced a prediction;
- `invalid_assigned_code_rate`: invalid assigned codes divided by assigned primary codes;
- `invalid_relation_rate`: invalid relation edges divided by returned relation edges;
- `validation_error_count` and `validation_error_kinds`: structural, offset, code-membership, and
  relation failures grouped for diagnosis;
- `missing_prediction_count`: documents for which the runtime returned no prediction.

An output with a high F1 but a failed invariant is not a valid promotion candidate. The validator
uses the same terminology release selected by the pipeline profile; it does not infer membership
from benchmark gold labels.

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

For the checked-in pilot, the reproducible handoff wrapper is:

```bash
bash scripts/review_vi_clinical_grounding_v1.sh \
  artifacts/review-packs/vi-clinical-grounding-v1
```

The generated directory is intentionally ignored by Git. Keep its manifest and edited reviewer
files in coordinator-controlled storage; publish only the resulting text-free agreement report
and dataset fingerprints when a release is authorized.

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
agreement versus `needs_adjudication`. Reviewer forms must set `review_complete: true`; an
untouched form cannot become an agreement by accident. It is deliberately not a gold dataset.
After an adjudicator resolves every item, use `review-snapshot-freeze` to export a separate
fingerprinted reviewed snapshot. The command emits the JSONL snapshot, a text-free
`review-agreement.json`, a derived `dataset_manifest.yaml`, and a checksum manifest. The default
policy requires double-review agreement or explicit adjudicator decisions. The source benchmark
remains immutable, and a synthetic pilot remains ineligible for clinical claims even after review
until a licensed clinical snapshot supplies that evidence.

CI also creates a fully double-assigned handoff for the generated 200-document diagnostic test
split. Download `public-benchmark-<commit>` from the CI run and use
`vi-clinical-grounding-synthetic-v1/review-pack/`. Both reviewer files contain all 200 documents,
while `coordinator_document_map.jsonl` remains separate. This artifact is review-ready but still
contains no human labels and makes no clinical claim.

For engineering QA, the repository also provides an independent technical contract review:

```bash
python scripts/review_vi_clinical_synthetic.py \
  --benchmark artifacts/benchmarks/vi-clinical-grounding-synthetic-v1 \
  --split test \
  --output artifacts/benchmarks/vi-clinical-grounding-synthetic-v1/technical-review.json
```

This review is intentionally not routed through the human review-pack importer. It validates every
document against an independently maintained template contract, including exact raw offsets,
type/assertion combinations, fixture code ownership, duplicate IDs, relation endpoints, and
context cues. The report contains hashes, counts, template groups, and failure codes but no raw
text. A passing technical review is useful for reproducibility and generator regression testing;
it does not change `eligible_for_clinical_claim`, which remains false for synthetic data. The
separate engineering gate may be true when all reproducibility and structural checks pass.
