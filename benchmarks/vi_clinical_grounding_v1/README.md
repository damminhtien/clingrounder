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

## Dataset policy

- Current status: synthetic pilot, version `0.1.0`.
- No private competition data, leaked data, or hosted model output is included.
- Validation and test records are kept separate from training records in the manifest.
- Near-duplicate and template leakage checks are required before a clinical release.
- A clinical release requires a human-reviewed status, reviewer agreement metadata, and complete
  review coverage for every declared split.

See [methodology](../../docs/benchmarks/vi_clinical_grounding_v1/methodology.md) for promotion
rules and the limitations of this pilot.
