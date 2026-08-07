# Phase 1 Benchmark Tests

This directory mirrors `clingrounder.benchmarks.phase1`. Tests collected here
receive the `benchmark` marker automatically and run in the nightly/full suite:

```bash
pytest -o addopts='' -m "benchmark and not private and not model" \
  tests/benchmarks/phase1
```

The default PR suite excludes them so core clinical NLP contracts remain fast.
Tests that need ignored manual gold or source documents also carry `private`.
