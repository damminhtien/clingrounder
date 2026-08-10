# Benchmark Expansion Plan

`vi-clinical-grounding-v1` is intentionally published as a tiny synthetic pilot in the current
release. It proves the runner, neutral schema, output contract, reproducibility checks, and
performance reporting; it does not establish clinical model quality.

## Promotion stages

| Stage | Dataset | Required evidence | Status |
| --- | --- | --- | --- |
| Pilot | Synthetic, checked-in | Offset/schema/reproducibility fixtures | Published |
| Public v1 | Redistributable Vietnamese notes, 500+ documents | License record, human review, held-out split | Not available yet |
| Challenge | Source- and template-held-out notes | Independent review and frozen manifest | Not available yet |

The 500+ target is a release gate, not a claim about the current repository. Expanding the fixture
without human review would make the benchmark look larger while weakening its scientific value.

## Reproducible synthetic expansion

For pipeline development, generate a larger synthetic snapshot outside the checked-in pilot:

```bash
uv run python scripts/generate_vi_clinical_benchmark.py \
  --output-dir /tmp/vi-clinical-grounding-synthetic-v1
```

The default creates `600/100/200` train/validation/test documents with disjoint template groups,
stable seed `42`, all five supported entity types, explicit lab `HAS_VALUE` relations,
raw-offset assertions, and a content-addressed manifest. Its status is
`synthetic_pending_human_review`; it must not be used for a clinical claim or silently replace the
published pilot. A reviewer can promote a later snapshot only after checking the generated cases,
adding the review record, and rerunning the source/license gates.

Generator version `0.2.0` selects concepts by semantic role rather than shuffled list position,
derives assertion labels from explicit template cues, and uses split-specific surface forms. CI
regenerates the default snapshot, audits normalized-text/template leakage, and verifies the
correctness and provenance reference. This protects engineering evidence but does not substitute
for independent human review.

## Required snapshot contract

Every future snapshot must include:

- a content-addressed dataset manifest and source/license dossier;
- document-level train/dev/test grouping, with near-duplicate and template leakage checks;
- raw text plus `[start, end)` offsets into that exact text;
- entity type, assertion, candidate, and relation annotations where available;
- reviewer roles, adjudication policy, agreement metrics, and unresolved-label policy;
- a machine-readable schema version and a deterministic build command.

Restricted manual gold and competition data may be used for local diagnostics, but cannot be copied
into this public benchmark. Sources that do not permit redistribution remain external snapshots
identified by fingerprints only.

## How to add a source

1. Add a source dossier under `docs/mining-sources/` and a registry entry with license/access and
   version information.
2. Acquire the source through its connector into the content-addressed artifact store.
3. Parse into neutral documents without rewriting the source text.
4. Run deduplication and leakage checks before annotation split assignment.
5. Export a reviewed benchmark snapshot and update the dataset manifest and methodology.
6. Run all benchmark configurations and record correctness, runtime, environment, model, and
   terminology fingerprints.

Until those gates pass, keep the data in mining/research layers and label the benchmark as a pilot.
