# API Stability

The package has one stable application entry point and one explicit advanced namespace.

## Stable API

Use the root package for application code:

```python
from medical_kg_nlp import Pipeline

with Pipeline.from_profile("clinical-baseline") as pipeline:
    prediction = pipeline.predict("Bệnh nhân khó thở.", document_id="note-001")
```

The root package owns the supported facade and immutable prediction contracts. It does not expose
factory internals or optional model implementations.

## Advanced API

Use `medical_kg_nlp.pipeline.advanced` when an integration must inject components, construct a
runtime, or work with stage-level traces:

```python
from medical_kg_nlp.pipeline.advanced import PipelineComponents, PipelineRunner

runner = PipelineRunner(PipelineComponents(...))
```

`medical_kg_nlp.pipeline` forwards to that same namespace. It is a forwarding import path, not a
second implementation. New advanced symbols must be added to `pipeline/advanced.py` only.

## Ownership rules

- `medical_kg_nlp.Pipeline` owns resources and lifecycle for ordinary callers.
- `PipelineFactory` is the composition root for advanced callers.
- `PipelineRunner` orchestrates already-created components; it does not load config or resources.
- Benchmark-specific APIs remain under `medical_kg_nlp.benchmarks`.
- Optional model dependencies are imported lazily by their adapters.

When removing an API, delete the implementation and its exports together, then update the migration
documentation and tests in the same change. Do not keep a compatibility copy of a runner, factory,
or retrieval path.
