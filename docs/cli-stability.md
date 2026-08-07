# CLI Stability Policy

The package exposes three command-line entry points with different support guarantees.

| Entry point | Scope | Support contract |
| --- | --- | --- |
| `clingrounder` | operational | Stable runtime, terminology, KG, evaluation, validation, and release commands. |
| `clingrounder-research` | research | Reproducible mining, model, and experiment workflows; APIs may change between minor releases. |
| `clingrounder-benchmark` | benchmark | Optional benchmark plugins and competition-specific workflows; no reusable-runtime stability promise. |

Operational imports do not load model, data, or benchmark modules. Optional dependencies are loaded
only when the selected command is invoked. A missing optional extra must produce an actionable
installation error rather than a partial fallback.

## Stability Rules

- Stable operational commands preserve their output schema and exit-code behavior within a minor
  release.
- Research commands are versioned by their config and artifact fingerprints, not by undocumented
  parser aliases.
- Benchmark commands are isolated from the reusable package and may be removed with the benchmark
  plugin that owns them.
- New commands belong to the narrowest scope that owns their resources.
- Unknown commands fail without importing model-heavy modules.
- The package does not maintain duplicate handlers for old and new command paths.

## Common Commands

```bash
clingrounder pipeline run --config configs/pipeline/clinical-baseline.yaml
clingrounder terminology inspect --index .cache/clingrounder/terminology/<index>.sqlite3
clingrounder validate --help

clingrounder-research data registry validate --help
clingrounder-research model inspect-token-classifier-run --help

clingrounder-benchmark list
clingrounder-benchmark phase1 --help
```

Use `clingrounder --help`, `clingrounder-research --help`, or `clingrounder-benchmark --help` to inspect
the supported surface for the installed package.
