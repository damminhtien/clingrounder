# Evidence And Release Checklist

ClinGrounder has three different release surfaces. They must not be promoted by the same
evidence or described with the same claims.

## 1. Runtime Release

This gate proves that the package is installable and deterministic. It does **not** prove clinical
quality.

- [ ] `ruff`, `mypy`, and the fast test suite pass.
- [ ] The wheel installs in a clean virtual environment and the installed-wheel smoke test passes.
- [ ] Core validation reports zero schema, offset, code-system, and artifact-integrity errors.
- [ ] The bundled resource manifest and every external resource manifest have stable SHA-256 values.
- [ ] A fixed-profile run can be reproduced from the recorded commit, lockfile, config, and resource
      fingerprints.
- [ ] The changelog describes breaking API or resource changes.

Required verification:

```bash
uv run ruff check .
uv run mypy src
uv run pytest -q
uv run clingrounder release audit \
  --policy configs/repository/public-release.yaml \
  --root .
```

## 2. Public Benchmark Evidence

This gate proves that a public dataset and its evaluation procedure are inspectable. A synthetic
fixture can validate engineering behavior, but it cannot support a clinical performance claim.

- [ ] Dataset source, license, version, split policy, and SHA-256 are declared.
- [ ] Training, validation, and test documents are disjoint by document and normalized text.
- [ ] Template or article leakage checks pass.
- [ ] Annotation structure and raw offsets validate without repair.
- [ ] Human review coverage and agreement are measured from independent review submissions.
- [ ] The test split is frozen before model or rule selection.
- [ ] The report names the exact config, commit, terminology fingerprints, and environment.
- [ ] `eligible_for_clinical_claim` is derived by the audit; it is never set by editing a boolean.

For the checked-in Vietnamese pilot:

```bash
clingrounder-benchmark suite \
  --benchmark benchmarks/vi_clinical_grounding_v1 \
  --config exact=configs/benchmarks/vi_clinical_grounding_v1/exact.yaml \
  --config lexical=configs/benchmarks/vi_clinical_grounding_v1/lexical.yaml \
  --config hybrid=configs/benchmarks/vi_clinical_grounding_v1/hybrid.yaml \
  --config full=configs/benchmarks/vi_clinical_grounding_v1/full.yaml \
  --output artifacts/benchmarks/vi-clinical-grounding-v1/suite

clingrounder-benchmark audit \
  --benchmark benchmarks/vi_clinical_grounding_v1 \
  --output artifacts/benchmarks/vi-clinical-grounding-v1/audit.json
```

The current pilot is intentionally synthetic and reports `eligible_for_clinical_claim: false`.
Its perfect fixture metrics are contract smoke results, not evidence that the toolkit generalizes
to clinical notes.

## 3. Model Or Resource Release

A model release requires more than a successful training job or a public checkpoint URL.

- [ ] The model ID, immutable revision, artifact SHA-256, tokenizer, and runtime profile are pinned.
- [ ] Training data manifests include source, license, processing code, split, and provenance.
- [ ] Restricted or hosted-only data is excluded from redistributable artifacts.
- [ ] The model card states intended use, excluded use, training data, evaluation protocol, and
      known limitations.
- [ ] Evaluation uses a human-reviewed, redistributable, source-separated public test snapshot.
- [ ] Results include per-type entity metrics, exact offsets, assertions, linking, abstention, and
      error slices rather than one aggregate number only.
- [ ] A rollback artifact and deterministic inference command are recorded.
- [ ] The governance contract is `ready`; `pending_public_snapshot` must remain pending otherwise.

The current model contract is deliberately not ready. VietBioNER and other mined sources remain
valuable diagnostic or pretraining inputs, but they do not by themselves provide the assertion,
linking, and clinical-note evidence required for a ClinGrounder model claim.

## Claim Vocabulary

Use these terms consistently:

| Term | Permitted claim |
| --- | --- |
| Runtime smoke | The package starts, validates offsets, and produces deterministic output for the fixture. |
| Synthetic pilot | The benchmark exercises schemas, metrics, and selected scenarios. |
| Research result | A measured result for a named dataset/config, with limitations stated. |
| Clinical validation | Only for a human-reviewed, licensed, source-separated public or authorized test set with an explicit protocol. |

Never describe a synthetic pilot, a mined silver layer, a competition artifact, or a local manual
gold set as clinical validation. The release audit is the final publication boundary; this document
is a checklist and does not override it.
