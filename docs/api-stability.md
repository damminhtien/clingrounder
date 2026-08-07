# API Stability

The package has one stable application entry point and one explicit advanced namespace.

## Stable API

Use the root package for application code:

```python
from clingrounder import Pipeline

with Pipeline.from_profile("clinical-baseline") as pipeline:
    prediction = pipeline.predict("Bệnh nhân khó thở.", document_id="note-001")
```

The root package owns the supported facade and immutable prediction contracts. It does not expose
factory internals or optional model implementations.

## Advanced API

Use `clingrounder.pipeline.advanced` when an integration must inject components, construct a
runtime, or work with stage-level traces:

```python
from clingrounder.pipeline.advanced import PipelineComponents, PipelineRunner

runner = PipelineRunner(PipelineComponents(...))
```

`clingrounder.pipeline` forwards to that same namespace. It is a forwarding import path, not a
second implementation. New advanced symbols must be added to `pipeline/advanced.py` only.

## Ownership rules

- `clingrounder.Pipeline` owns resources and lifecycle for ordinary callers.
- `PipelineFactory` is the composition root for advanced callers.
- `PipelineRunner` orchestrates already-created components; it does not load config or resources.
- Benchmark-specific APIs remain under `clingrounder.benchmarks`.
- Optional model dependencies are imported lazily by their adapters.

When removing an API, delete the implementation and its exports together, then update the migration
documentation and tests in the same change. Do not keep a compatibility copy of a runner, factory,
or retrieval path.

## Configuration Ownership

Reusable profiles group runtime policy by subsystem. The factory compiles these blocks into one
immutable runtime policy before constructing components:

```yaml
pipeline:
  context:
    provider: rules
    context_window: 80
  linking:
    provider: rules
    candidate_sources: [exact, abbreviation, bm25]
    reranker:
      provider: rules
  graph:
    provider: disabled
  relations:
    provider: disabled
    validate_with_kg: false
  validation:
    entities_with_kg: true
    relations_with_kg: false
  runtime:
    backend: serial
    workers: 1
```

Unknown nested keys fail before terminology or model resources are loaded. `compiled_options` in
config inspection is diagnostic output only; callers should edit the subsystem blocks above.
