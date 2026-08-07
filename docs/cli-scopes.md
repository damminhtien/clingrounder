# CLI Scopes

The installed commands share one lazy dispatcher and one handler registry, but expose different
responsibilities:

| Command | Scope | Contents |
| --- | --- | --- |
| `clingrounder` | operational | pipeline execution, terminology, graph operations, validation, release checks |
| `clingrounder-research` | research | data mining and local model training |
| `clingrounder-benchmark` | benchmark | optional benchmark plugins and promotion comparisons |

No handler is copied between entrypoints. The scope only controls parser registration, so an
operational process cannot accidentally expose benchmark or training commands. Optional model and
benchmark modules remain lazy imports.

Examples:

```bash
clingrounder pipeline run --config configs/pipeline/clinical-baseline.yaml \
  --input data/samples/sample_notes.jsonl --output outputs/predictions.jsonl

clingrounder-research data registry validate
clingrounder-research model inspect-token-classifier-run --config configs/benchmarks/phase1/models/run.yaml
clingrounder-benchmark list
```

Library tests may call `clingrounder.cli.main.main()` without a scope to construct the complete
parser. Installed applications should use one of the scoped entrypoints above.
