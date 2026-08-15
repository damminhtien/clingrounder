# Release Status

This page is the concise release-readiness view for ClinGrounder. It separates software
correctness from evidence for clinical performance; a passing package or benchmark smoke test
does not promote a clinical claim.

## Current State

| Surface | Status | Meaning |
| --- | --- | --- |
| Python package and deterministic runtime | Ready for alpha release | The package, bundled resource pack, validation, and offline wheel smoke path are reproducible. |
| Public engineering benchmark | Ready as synthetic pilot | The schema, split checks, metrics, artifacts, and runner are public and auditable. |
| Vietnamese model release | Pending public snapshot | The training contract is inspectable, but no model is promoted as a public clinical model. |
| Clinical performance claim | Not eligible | The current benchmark source is synthetic and has no independent clinical review. |

## Evidence Commands

Run these commands from a clean checkout:

```bash
uv run ruff check .
uv run mypy src
uv run pytest -q -o addopts=''
uv run clingrounder release audit \
  --policy configs/repository/public-release.yaml \
  --root .
uv run clingrounder-benchmark audit \
  --benchmark artifacts/benchmarks/vi-clinical-grounding-synthetic-v1/dataset \
  --output /tmp/clingrounder-dataset-audit.json
uv run clingrounder-research model inspect-public-training-contract \
  --config configs/training/vi_clinical_ner_v1.yaml
```

The current synthetic audit is expected to report:

```text
eligible_for_engineering_use: true
eligible_for_clinical_claim: false
clinical_claim_blockers:
  - synthetic_source
  - human_review_required
  - release_status_not_reviewed
```

These are deliberate governance results, not test failures.

## Promotion Path

`eligible_for_clinical_claim` is derived by the dataset audit. It must never be changed by
editing a manifest boolean or by treating an internal technical review as clinical review.

To create an eligible clinical evidence snapshot, the project needs all of the following:

1. A redistributable or explicitly authorized non-synthetic clinical source with a pinned version,
   license, access class, and immutable split fingerprints.
2. Independent clinical reviewers who annotate the gold-blind review pack, with reviewer identity
   and review protocol recorded outside the prediction files.
3. Double-review agreement and adjudication meeting the configured targets for spans, assertions,
   and relations.
4. A source-held-out and template-held-out test split frozen before model or rule selection.
5. A model artifact manifest, if a model is released, containing immutable model/tokenizer
   revisions, artifact SHA-256, training snapshot fingerprints, intended use, and limitations.

Only after those inputs are present should the dataset audit produce `eligible_for_clinical_claim:
true` and the training contract move from `pending_public_snapshot` to `ready`.

## Review Handoff

Generate a deterministic gold-blind pack for an authorized reviewer:

```bash
uv run clingrounder-benchmark review-pack \
  --benchmark <licensed-dataset-directory> \
  --split test \
  --reviewer clinical-reviewer-1 \
  --reviewer clinical-reviewer-2 \
  --double-review-fraction 0.10 \
  --output /path/to/review-pack
```

The checked-in synthetic pilot can exercise this workflow, but it remains ineligible even after
review because synthetic text is not clinical validation. The review pack must never be committed
with completed labels or raw restricted text.

## Claim Vocabulary

- **Runtime smoke:** the package starts, validates, and produces deterministic output for a
  fixture.
- **Synthetic pilot:** a public engineering fixture for schemas, metrics, and reproducibility.
- **Research result:** a measured result tied to a named dataset, configuration, and artifact
  fingerprint.
- **Clinical validation:** an independently reviewed, authorized or redistributable,
  source-separated clinical test snapshot with a documented protocol.

Do not use the synthetic pilot, mined silver data, competition artifacts, or local manual gold as
clinical validation.
