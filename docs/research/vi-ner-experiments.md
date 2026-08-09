# Vietnamese NER Research Track

This page records the Vietnamese token-classification experiments that informed ClinGrounder's
model-release policy. The runs are research evidence, not a shipped model and not clinical
validation. Checkpoints and source snapshots stay outside the public wheel; the local manifests
are the authority for restoring them.

## Task And Data

The experiments use the five-type span contract:

```text
DISEASE, SYMPTOM, DRUG, LAB_TEST, LAB_RESULT
```

The strongest directly comparable run used 155 records and 3,198 entities: 135 train records and
20 development records. The training snapshot, source labels, and split fingerprints are recorded
in the run manifests under `outputs/models/` when the research artifacts are restored. The
competition/manual-gold source is intentionally not treated as a public clinical dataset.

## Measured Runs

| Run | Model / initialization | Dev span precision | Dev span recall | Dev span F1 | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `phase1-five-type-xlmr-qa-edu-e12-lr5e5-2026-07-27` | XLM-R base fine-tune | 0.6854 | 0.7669 | **0.7239** | baseline |
| `phase1-five-type-xlmr-dapt-qa-edu-2026-07-29` | XLM-R + joint DAPT initialization | 0.6800 | 0.7529 | 0.7146 | rejected |

The DAPT-specialized run regressed exact-span F1 by `0.0093`, increased false positives from 151
to 152, and increased false negatives from 100 to 106 on the frozen development split. The
promotion gate therefore rejected it. This is useful evidence: domain-adaptive pretraining is not
promoted merely because its training loss is lower or its corpus is larger.

## DAPT Corpus And Objective

The joint DAPT run used separate provenance lanes:

| Lane | Records | Objective |
| --- | ---: | --- |
| VietBioNER unlabeled | 63 | masked language modeling |
| VietMed-NER unlabeled | 9,246 | masked language modeling |
| Round 2 unlabeled | 100 | masked language modeling only |

It also used 63,247 terminology synonym pairs for the contrastive objective. The run manifest
records that Round 2 was excluded from entity supervision, pseudo-labeling, synonym-pair creation,
and threshold calibration. The trained XLM-R base has 278,295,186 parameters. Reported training
losses were MLM `1.7907` and synonym contrastive `0.3603`; these are optimization diagnostics,
not task quality metrics.

## Reproduction

Restore the referenced local snapshots and run the pinned configuration with the research extra:

```bash
uv sync --frozen --extra ml
uv run clingrounder-research model inspect-token-classifier-run \
  --config configs/benchmarks/phase1/models/phase1-five-type-xlmr-qa-edu-e12-lr5e5-2026-07-27.yaml

uv run clingrounder-research model inspect-token-classifier-run \
  --config configs/benchmarks/phase1/models/phase1-five-type-xlmr-dapt-qa-edu-2026-07-29.yaml
```

The manifests pin the Git commit, model revision, dataset SHA-256, configuration SHA-256,
lockfile SHA-256, device, precision, seed, and evaluation split. A clean machine must restore the
same snapshots before attempting a training or inference run; no hidden download is performed.

## Release Boundary

These runs do not make the public model contract `ready` because they do not provide a
redistributable, independently human-reviewed, source-separated clinical test snapshot. Until
that gate is met:

- no checkpoint is loaded by the default deterministic pipeline;
- no local competition result is presented as clinical performance;
- no restricted training snapshot is copied into the package or resource pack;
- the rule-based pipeline remains the stable public baseline.

See [model-training.md](../model-training.md), [evidence-release-checklist.md](../evidence-release-checklist.md),
and [research-portfolio.md](../research-portfolio.md) for the distinction between a research run,
a public artifact, and a clinical claim.
