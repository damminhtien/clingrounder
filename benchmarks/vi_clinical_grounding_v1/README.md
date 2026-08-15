# Vietnamese Clinical Grounding v1

This directory defines the public benchmark contract for ClinGrounder. The checked-in fixture is
a small, redistributable synthetic **pilot** used to verify the runner, schema, metrics, and
reproducibility workflow. It is not a clinical validation study and must not be presented as one.

## Run

From a source checkout:

```bash
clingrounder-benchmark suite \
  --benchmark benchmarks/vi_clinical_grounding_v1 \
  --config exact=configs/benchmarks/vi_clinical_grounding_v1/exact.yaml \
  --config lexical=configs/benchmarks/vi_clinical_grounding_v1/lexical.yaml \
  --config hybrid=configs/benchmarks/vi_clinical_grounding_v1/hybrid.yaml \
  --config full=configs/benchmarks/vi_clinical_grounding_v1/full.yaml \
  --output artifacts/benchmarks/vi-clinical-grounding-v1/suite
```

The command writes deterministic JSON/Markdown artifacts for each profile and one suite-level
ablation table. A later human-reviewed release can replace the pilot data without changing the
output contract. Use `clingrounder-benchmark run` when you need only one profile.

Audit the dataset before treating its results as public evidence:

```bash
clingrounder-benchmark audit \
  --benchmark benchmarks/vi_clinical_grounding_v1 \
  --output artifacts/benchmarks/vi-clinical-grounding-v1/audit.json
```

Add `--strict` in CI or a release workflow. The current synthetic pilot intentionally returns
`eligible_for_clinical_claim: false`; this is a governance result, not a runner failure. The audit
checks declared SHA-256 values, document IDs, normalized text leakage, template overlap, split
policy, license metadata, and human-review coverage without writing document text to the report.

Compare a candidate summary with a previous run using the same task-neutral promotion contract:

```bash
clingrounder-benchmark dataset-compare \
  --baseline artifacts/benchmarks/baseline/summary.json \
  --candidate artifacts/benchmarks/candidate/summary.json \
  --policy benchmarks/vi_clinical_grounding_v1/expected_results.yaml \
  --output artifacts/benchmarks/candidate/promotion.json
```

The gate requires the configured primary improvement, protects linking/assertion metrics and
latency, and fails closed on invalid offsets, terminology assignments, relations, or validation
errors. Runtime-only comparison remains available as `clingrounder-benchmark compare`.

Verify that a generated suite still matches the correctness values published in the benchmark
reference file. New references also pin the dataset, profile, and terminology fingerprints. The
command reports p95 latency for comparison but deliberately does not gate on it because latency
depends on the host:

```bash
clingrounder-benchmark verify-reference \
  --suite artifacts/benchmarks/vi-clinical-grounding-v1/suite/suite.json \
  --reference benchmarks/vi_clinical_grounding_v1/expected_results.yaml \
  --output artifacts/benchmarks/vi-clinical-grounding-v1/reference-verification.json
```

The larger generated diagnostic has a separate reference at
`synthetic_diagnostic_expected_results.yaml`. It remains synthetic and review-pending; CI uses it
to detect generator or pipeline drift, not to support a clinical claim.

Reproduce that larger snapshot locally with the repository wrapper:

```bash
bash scripts/reproduce_vi_clinical_grounding_expanded.sh \
  artifacts/benchmarks/vi-clinical-grounding-synthetic-v1
```

The default snapshot contains 600 train, 100 validation, and 200 test documents generated from
split-disjoint templates. The command writes the generated dataset, split audit, per-profile
benchmark artifacts, suite report, and reference verification under the supplied directory. Set
`CLINGROUNDER_BIN` or `PYTHON_BIN` when using a non-default environment. The generated records are
synthetic diagnostic fixtures and remain ineligible for clinical claims even after technical
review; only a separately sourced, human-reviewed licensed clinical snapshot can satisfy that
gate.

Run the independent technical review for the generated test split:

```bash
python scripts/review_vi_clinical_synthetic.py \
  --benchmark artifacts/benchmarks/vi-clinical-grounding-synthetic-v1 \
  --split test \
  --output artifacts/benchmarks/vi-clinical-grounding-synthetic-v1/technical-review.json
```

The review checks all documents against the declared seven-template contract, raw offset/text
ownership, entity type and assertion values, fixture terminology membership, duplicate IDs, and
relation endpoints. Its report is PHI-safe and records `reviewer: codex`,
`review_kind: template_and_invariant_review`, and `human_clinical_review: false`. It is an
engineering QA artifact. A passing report may be used as `eligible_for_engineering_use: true`,
but it is not a human annotation or clinical claim.

## Independent review handoff

Create a deterministic gold-blind pack before asking reviewers to annotate a future licensed
snapshot:

```bash
bash scripts/review_vi_clinical_grounding_v1.sh
```

The script is a convenience wrapper around the CLI and accepts an optional output directory as
its first argument. It uses a fixed seed and reviewer assignment policy so another machine can
recreate the same handoff.

Each reviewer directory contains only source text, a stable `review_id`, safe display metadata,
and empty annotation arrays. Gold annotations and source document IDs stay in the coordinator
mapping. The generated manifest records source fingerprints, assignment counts, and the seed. The
pack is a handoff artifact, not evidence that the pilot has been human-reviewed.

After reviewers finish editing their assigned `items.jsonl` files, set
`review_complete: true` on every completed item, then validate and join the submissions into an
adjudication queue:

```bash
clingrounder-benchmark review-pack-import \
  --benchmark benchmarks/vi_clinical_grounding_v1 \
  --pack artifacts/review-packs/vi-clinical-grounding-v1 \
  --split test \
  --output artifacts/review-imports/vi-clinical-grounding-v1
```

The importer verifies source and assignment fingerprints, raw offsets, entity taxonomy,
code-system shape, relation endpoints, and reviewer completeness. It marks exact agreement and
disagreement in `adjudication.jsonl`; it never promotes annotations to gold automatically.

After a human adjudicator resolves every `needs_adjudication` row by adding
`status: adjudicated`, `adjudicated_entities`, and `adjudicated_relations`, freeze a separate
reviewed snapshot:

```bash
clingrounder-benchmark review-snapshot-freeze \
  --benchmark benchmarks/vi_clinical_grounding_v1 \
  --import-dir artifacts/review-imports/vi-clinical-grounding-v1 \
  --output artifacts/reviewed-snapshots/vi-clinical-grounding-v1 \
  --split test
```

The freeze command requires double-review agreement or explicit adjudication by default. It
writes four portable artifacts: the fingerprinted JSONL snapshot, a text-free
`review-agreement.json`, a derived `dataset_manifest.yaml`, and a top-level `manifest.json` with
checksums for all three. The derived dataset manifest marks synthetic review output as
`synthetic_reviewed`, so review improves annotation provenance without making the pilot clinical
evidence. The command does not mutate this benchmark directory or change `human_reviewed` by
editing YAML.

## Dataset policy

- Current status: synthetic pilot, version `0.1.0`.
- No private competition data, leaked data, or hosted model output is included.
- Validation and test records are kept separate from training records in the manifest.
- Near-duplicate and template leakage checks are required before a clinical release.
- The pilot includes one synthetic `LAB_TEST -> LAB_RESULT` relation so relation extraction and
  endpoint validation are exercised; this is an engineering fixture, not clinical evidence.
- A clinical release requires a human-reviewed status, complete review coverage for every
  declared split, and a separately hashed ``clingrounder.review-agreement.v1`` report whose
  measured metrics meet the manifest targets. Merely setting ``human_reviewed: true`` is not
  sufficient.

See [methodology](../../docs/benchmarks/vi_clinical_grounding_v1/methodology.md) for promotion
rules and the limitations of this pilot.
