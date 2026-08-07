# Vietnamese Clinical Grounding v1

This directory defines the public benchmark contract for ClinGrounder. The checked-in fixture is
a small, redistributable synthetic **pilot** used to verify the runner, schema, metrics, and
reproducibility workflow. It is not a clinical validation study and must not be presented as one.

## Run

From a source checkout:

```bash
clingrounder-benchmark run \
  --benchmark benchmarks/vi_clinical_grounding_v1 \
  --config configs/benchmarks/vi_clinical_grounding_v1/full.yaml \
  --output artifacts/benchmarks/vi-clinical-grounding-v1/full
```

The command writes deterministic JSON/Markdown artifacts. A later human-reviewed release can
replace the pilot data without changing the output contract.

## Dataset policy

- Current status: synthetic pilot, version `0.1.0`.
- No private competition data, leaked data, or hosted model output is included.
- Validation and test records are kept separate from training records in the manifest.
- Near-duplicate and template leakage checks are required before a clinical release.

See [methodology](../../docs/benchmarks/vi_clinical_grounding_v1/methodology.md) for promotion
rules and the limitations of this pilot.
