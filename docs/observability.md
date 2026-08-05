# Pipeline Observability

The pipeline keeps inference independent from telemetry vendors. `PipelineObserverPort`
receives stage lifecycle events and bounded counters. The default observer is a no-op;
`InMemoryPipelineObserver` is suitable for research runs and tests.

Traces hash document identifiers and do not include raw note text by default. Error messages
are redacted by default; enable `PipelineComponents.trace_include_error_messages` only for a
controlled local debugging run. Configuration, terminology, and model fingerprints are
recorded as low-cardinality metadata.

Stage records also carry outcome (`success`, `failure`, or `cancelled`), start time,
elapsed time, queue wait, bounded error type/message, document length, backend and worker.
The in-memory snapshot exposes document/entity counts, assignment coverage, abstention,
candidate-count distribution, validation failures by type, and model forward-pass counts.
Trace retention is bounded; a failed document retains its partial trace on the raised
exception as `pipeline_trace`.

OpenTelemetry is optional and lazy-loaded:

```python
from medical_kg_nlp.pipeline import OpenTelemetryPipelineObserver, PipelineComponents

observer = OpenTelemetryPipelineObserver()
components = PipelineComponents(..., observer=observer)
```

Install the OpenTelemetry SDK/exporter separately. The core package remains importable without
it, and observer failures are swallowed so telemetry cannot change prediction behavior.
