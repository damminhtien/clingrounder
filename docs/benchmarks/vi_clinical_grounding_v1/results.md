# Benchmark Results

This page separates the checked-in wiring pilot from the larger generated diagnostic snapshot.
Neither result is a clinical performance claim: the repository does not yet contain a
human-reviewed public clinical test set.

## Checked-In Pilot

The checked-in `0.1.0` fixture contains four synthetic test documents, including one
`LAB_TEST -> LAB_RESULT` relation. The suite runner was executed on commit `ae7de75` in GitHub CI
on Ubuntu with Python 3.14. The relation extractor is enabled only by the full profile. The
numbers below are a fresh reference run; latency is machine-dependent and is not a portability
claim. Regenerate the JSON artifact bundle instead of treating these values as a universal speed
target.

| System | Entity exact F1 | Assertion macro-F1 | Recall@5 | Top-1 | Relation F1 | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact dictionary | 1.0000 | 1.0000 | 0.8750 | 0.8750 | 0.0000 | 7.263 |
| Lexical | 1.0000 | 1.0000 | 0.8750 | 0.8750 | 0.0000 | 2.355 |
| Hybrid | 1.0000 | 1.0000 | 0.8750 | 0.8750 | 0.0000 | 3.592 |
| Full deterministic | 1.0000 | 1.0000 | 0.8750 | 0.8750 | 1.0000 | 3.704 |

Only the full profile predicts the single gold relation because the other profiles intentionally
disable relation extraction. These values verify the output contract and are not useful for
ranking extraction systems on clinical data.

The test split contains 9 entities, 8 with gold terminology codes. `Recall@k` and MRR use all 8
linkable gold entities as their denominator, including entities missed by extraction; assignment
coverage is the fraction of predicted entities with an assigned primary code (`7/9` in this run).
This distinction prevents NER misses from disappearing from linking metrics.

## Generated Expansion Diagnostic

The reproducible generator was run with seed `42` and default sizes of 600 train, 100 validation,
and 200 test documents. The generated snapshot is explicitly marked
`synthetic_pending_human_review`; it is not checked into the repository and must not be used as
clinical evidence.

Run fingerprints:

| Split | Documents | SHA-256 |
| --- | ---: | --- |
| Train | 600 | `9281dadb0d4ba2bdc28b6d7b5dedd01195b2f429a73193fc73153e463586f3ef` |
| Validation | 100 | `e025c8e40e8bb60d3a500a352509ceae13cc6a8aae53e588124bbfe77d5369ca` |
| Test | 200 | `a4a71d40825fe2b3db3010d3b43dd0c7a2d36e480ac82f1abb12aeac0e1b444e` |

The dataset manifest SHA-256 is
`e2e4ea7b15804efdfeb5b5fe8b0eaeb9829a51a16b8d8db0faa23a8614ddb25a`.
The audit found zero structural or leakage issues. It reports
`eligible_for_engineering_use: true` for reproducible development and correctly reports
`eligible_for_clinical_claim: false` because no independent human review exists.
The machine-readable audit also reports the blockers `synthetic_source`,
`human_review_required`, and `release_status_not_reviewed`; these are release-governance
signals, not model-quality failures.

The latest reproducibility run was executed on commit `ae7de751f7fb2d36e02200ee2ee6ffde1eaf17d6`
using the same generated test split. Correctness metrics
are identical because these profiles change retrieval and graph behavior, not the rule proposal
vocabulary that dominates this snapshot. This is a regression/stress check, not evidence that the
retrieval variants are equivalent on real clinical text.

| System | Entity exact F1 | Assertion macro-F1 | Recall@5 | Top-1 | Relation F1 | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact dictionary | 0.5408 | 1.0000 | 0.3965 | 0.3965 | 0.0000 | 2.162 |
| Lexical | 0.5408 | 1.0000 | 0.3965 | 0.3965 | 0.0000 | 2.112 |
| Hybrid | 0.5408 | 1.0000 | 0.3965 | 0.3965 | 0.0000 | 2.936 |
| Full deterministic | 0.5408 | 1.0000 | 0.3965 | 0.3965 | 0.0000 | 3.010 |

The generated run had exact entity precision `1.0000`, recall `0.3706`, assignment coverage
`1.0000` over predicted entities, and zero output-validation errors. By type, the exact baseline
measured F1 `0.5546` for DISEASE, `0.2985` for DRUG, `0.6705` for SYMPTOM, and `0.0000` for both
LAB_TEST and LAB_RESULT. The positive assertion macro-F1 of `1.0000` is conditional on exact
entity matches; it does not compensate for missed entities. The absence of extracted lab
endpoints also explains relation F1 `0.0000` in the full profile despite 28 gold lab relations.

These values replaced an invalid earlier diagnostic whose shuffled concepts could receive the
wrong semantic role or assertion cue. The corrected generator intentionally exposes the bundled
pack's vocabulary and lab-recall limits instead of manufacturing an optimistic result. The
machine-readable reference is
`benchmarks/vi_clinical_grounding_v1/synthetic_diagnostic_expected_results.yaml`; CI regenerates
the 900-document snapshot and verifies its data, config, terminology, and correctness
fingerprints on every push.

The same CI job emits a gold-blind review pack for all 200 generated test documents with two
independent assignments per document. It is included under
`vi-clinical-grounding-synthetic-v1/review-pack/` in the `public-benchmark-<commit>` artifact.
Generating the handoff does not change the dataset's pending-review status; only completed,
validated reviewer submissions and explicit adjudication can produce a reviewed snapshot.

Runtime values are machine-dependent. The authoritative JSON artifacts contain initialization,
p50/p95/p99 latency, throughput, RSS, and model-forward counters; regenerate them with the
commands below instead of comparing numbers across machines.

## Reproduce

Checked-in pilot suite:

```bash
bash scripts/reproduce_vi_clinical_grounding_v1.sh \
  artifacts/benchmarks/vi-clinical-grounding-v1/suite
```

Generated diagnostic expansion:

```bash
uv run python scripts/generate_vi_clinical_benchmark.py \
  --output-dir /tmp/vi-clinical-grounding-synthetic-v1

uv run clingrounder-benchmark audit \
  --benchmark /tmp/vi-clinical-grounding-synthetic-v1 \
  --output /tmp/vi-clinical-grounding-synthetic-v1/audit.json

uv run clingrounder-benchmark suite \
  --benchmark /tmp/vi-clinical-grounding-synthetic-v1 \
  --config exact=configs/benchmarks/vi_clinical_grounding_v1/exact.yaml \
  --config lexical=configs/benchmarks/vi_clinical_grounding_v1/lexical.yaml \
  --config hybrid=configs/benchmarks/vi_clinical_grounding_v1/hybrid.yaml \
  --config full=configs/benchmarks/vi_clinical_grounding_v1/full.yaml \
  --output /tmp/vi-clinical-grounding-synthetic-v1/suite

uv run clingrounder-benchmark verify-reference \
  --suite /tmp/vi-clinical-grounding-synthetic-v1/suite/suite.json \
  --reference benchmarks/vi_clinical_grounding_v1/synthetic_diagnostic_expected_results.yaml
```

Before publishing a clinical result, replace the synthetic manifest with a licensed,
human-reviewed, source-held-out dataset and record its review and provenance metadata.
