# Promotion Benchmarks

The reusable runtime has a small, redistributable benchmark independent of any
competition corpus. It covers short and long notes, medication lists, lab
tables, negation, family history, noisy Vietnamese-English text, repeated
mentions, and candidate ambiguity.

Run it with a pinned profile:

```bash
clingrounder-benchmark runtime run \
  --input data/benchmarks/promotion/runtime_documents.jsonl \
  --config configs/pipeline/clinical-baseline.yaml \
  --output artifacts/benchmarks/runtime.json \
  --warmup 1 --repeats 5
```

Compare two JSON artifacts in CI:

```bash
clingrounder-benchmark compare baseline.json candidate.json \
  --output artifacts/benchmarks/comparison.json
```

The report records the commit, profile and terminology fingerprints, runtime
environment, initialization time, throughput, p50/p95/p99 latency, peak RSS,
stage latency, model forward-pass counters, candidate recall, assignment
coverage, and validation errors. Raw clinical text is never emitted.

Correctness gates are strict: offsets must be valid, assigned and relation
codes must be valid, output must be deterministic, and candidate ordering must
be stable. Candidate recall and positive assertion recall are protected from
regression. Timing and RSS use configurable tolerances because they depend on
the host. Run model/GPU benchmarks separately from the model-free CI job.

Benchmark JSON is an artifact, not a tracked dataset. Keep new fixtures
redistributable and do not add private or competition test data.
