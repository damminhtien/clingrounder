# Benchmark Results

This page separates the checked-in wiring pilot from the larger generated diagnostic snapshot.
Neither result is a clinical performance claim: the repository does not yet contain a
human-reviewed public clinical test set.

## Checked-In Pilot

The checked-in `0.1.0` fixture contains four synthetic test documents, including one
`LAB_TEST -> LAB_RESULT` relation. The suite runner was executed on commit `02cb7a9` on macOS with
Python 3.14. The relation extractor is enabled only by the full profile. The numbers below are a
fresh reference run; latency is machine-dependent and is not a portability claim. Regenerate the
JSON artifact bundle instead of treating these values as a universal speed target.

| System | Entity exact F1 | Assertion macro-F1 | Recall@5 | Top-1 | Relation F1 | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact dictionary | 1.0000 | 1.0000 | 0.8750 | 0.8750 | 0.0000 | 9.580 |
| Lexical | 1.0000 | 1.0000 | 0.8750 | 0.8750 | 0.0000 | 5.589 |
| Hybrid | 1.0000 | 1.0000 | 0.8750 | 0.8750 | 0.0000 | 5.983 |
| Full deterministic | 1.0000 | 1.0000 | 0.8750 | 0.8750 | 1.0000 | 6.867 |

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
| Train | 600 | `24c566e41967a6a3034d0f799ea465ee92a78094d3a23cc46e30dfc349dbcc2d` |
| Validation | 100 | `a4dad0b5bc27dd8fb26623b1fce91b82ef8b00f6395ac44e0dbd7aded9ae491e` |
| Test | 200 | `a6b74b1dd37a482efaf379b6609e893376d158ef5834657f419d053f96396192` |

All variants were run on commit `44c0aa3` using the same generated test split. Correctness metrics
are identical because the current profile differences do not change the generated entity
decisions. The bundled pack does not fully cover the generated lab concepts, so the snapshot is
also useful for exposing coverage gaps; runtime still varies by profile. This is a
regression/stress check, not evidence that retrieval variants are equivalent on real clinical
text.

| System | Entity exact F1 | Assertion macro-F1 | Recall@5 | Top-1 | Relation F1 | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact dictionary | 0.8390 | 0.2938 | 0.7023 | 0.7023 | 0.0000 | 4.35 |
| Lexical | 0.8390 | 0.2938 | 0.7023 | 0.7023 | 0.0000 | 4.31 |
| Hybrid | 0.8390 | 0.2938 | 0.7023 | 0.7023 | 0.0000 | 5.60 |
| Full deterministic | 0.8390 | 0.2938 | 0.7023 | 0.7023 | 0.8475 | 5.48 |

The generated run had exact entity recall `0.8511`, precision `0.8272`, assignment coverage
`0.8511`, and zero validation errors. By type, the exact baseline measured F1 `1.0000` for
DISEASE/DRUG, `0.8447` for SYMPTOM, `0.5814` for LAB_TEST, and `0.8366` for LAB_RESULT.
The largest diagnostic gaps are assertion positive macro-F1 `0.0000`, lab-test recall, and
linking coverage for the synthetic lab concepts that are intentionally absent from the small
pack. The `full` profile enables the deterministic relation extractor and KG validation, reaching
relation F1 `0.8475` on the generated lab relation; the other three profiles intentionally disable
relations. This is an architectural baseline, not evidence of clinical relation performance.
These failures are targets for human review and future model/resource work, not reasons to add
more unvalidated rules.

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

uv run clingrounder-benchmark run \
  --benchmark /tmp/vi-clinical-grounding-synthetic-v1 \
  --config configs/benchmarks/vi_clinical_grounding_v1/full.yaml \
  --output /tmp/vi-clinical-grounding-synthetic-v1/run \
  --split test
```

Before publishing a clinical result, replace the synthetic manifest with a licensed,
human-reviewed, source-held-out dataset and record its review and provenance metadata.
