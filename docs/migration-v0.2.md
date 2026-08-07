# Migration To 0.2

Version 0.2 is intentionally breaking. It separates reusable clinical NLP infrastructure from
rules, storage backends, experiments, and competition code. No compatibility import shims are
provided.

Pipeline profiles use the strict `clingrounder.pipeline-profile` envelope. Profiles written for
v1 must be rewritten explicitly before loading; the runtime does not guess or apply hidden
defaults for an unsupported profile version.

## Command Migration

| Removed root script | Replacement |
| --- | --- |
| `scripts/run_pipeline.py` | `clingrounder pipeline run` |
| `scripts/build_terminology_index.py` | `clingrounder terminology build` |
| `scripts/evaluate.py` | `clingrounder evaluate` |
| `scripts/validate_predictions.py` | `clingrounder validate` |
| `scripts/build_phase1_submission.py` | `clingrounder-benchmark phase1 submission` |

Run installed commands through `uv run`, for example:

```bash
uv run clingrounder pipeline run \
  --input data/samples/sample_notes.jsonl \
  --output outputs/predictions.jsonl

uv run clingrounder validate \
  --profile development \
  --pred outputs/predictions.jsonl \
  --documents data/samples/sample_notes.jsonl \
  --dictionary data/dictionaries/seed_concepts.jsonl
```

The Phase 1 command now takes explicit artifact inputs and policies. Legacy campaign-only flags
such as selective calibration, proposal sources, and run manifests stay in dedicated Phase 1
experiment scripts or package APIs instead of the stable CLI:

```bash
uv run clingrounder-benchmark phase1 submission \
  --input-dir data/raw/input \
  --output-dir outputs/phase1/current/output \
  --zip outputs/phase1/current/output.zip \
  --dictionary data/dictionaries/seed_concepts.jsonl \
  --assertion-policy empty \
  --candidate-policy empty
```

## Pipeline Construction

Before 0.2, callers could construct `PipelineRunner` with dictionaries, flags, or concrete stages.
The runner now accepts exactly one `PipelineComponents` object:

```python
from clingrounder.pipeline import PipelineComponents, PipelineRunner

runner = PipelineRunner(
    PipelineComponents(
        entity_extractor=my_extractor,
        assertion_classifier=my_assertion_adapter,
        options=my_options,
    )
)
```

For ordinary application code, use the managed facade:

```python
from clingrounder import Pipeline

with Pipeline.from_profile("clinical-baseline") as pipeline:
    prediction = pipeline.predict("Bệnh nhân khó thở.", document_id="note-001")
```

At advanced executable boundaries, use the composition root:

```python
from clingrounder.pipeline import PipelineFactory

runner = PipelineFactory.from_config(config_mapping)
```

`PipelineRunner` no longer reads YAML, opens dictionaries, or chooses rule/model implementations.
Every enabled option must have a corresponding injected port.

## Retrieval And Terminology

- `DictionaryStore` remains a canonical JSONL/in-memory utility and recognition vocabulary.
- Full normalization uses `TerminologyRepository` and a prebuilt `SQLiteTerminologyRepository`.
- Candidate generation is a `RetrievalPipeline` composed from retriever adapters.
- Runtime refuses missing or stale SQLite metadata. Build indexes explicitly before startup.
- Dense retrieval requires a separate `DenseVectorIndexPort`; no ANN backend is enabled by default.

Build the derived index before configuring `terminology.normalization_index_path`:

```bash
uv run clingrounder terminology build \
  --source data/processed/full_concepts.jsonl \
  --output .cache/clingrounder/terminology/full.sqlite3
```

## Model Adapters

Install optional local model dependencies with:

```bash
uv sync --extra dev --extra ml
```

Model config requires both `model_id` and `revision`. Loading is lazy and local-only. Token
classification output is projected through fast-tokenizer offsets and rejected if a final span does
not satisfy `source[start:end] == entity.text`.

## Evaluation Imports

Generic code stays under `clingrounder.evaluation`:

```python
from clingrounder.evaluation import EvaluationAdapter, evaluate_predictions
```

Move task and experiment imports as follows:

| Old ownership | 0.2 ownership |
| --- | --- |
| Phase 1 schemas, scoring, export, manual gold | `clingrounder.benchmarks.phase1` |
| Phase 1 pipeline report enrichment | `clingrounder.benchmarks.phase1.pipeline_report` |
| Ablations and trace aggregation | `clingrounder.experiments.ablation` |
| Loop engineer, journal, policy, artifacts | `clingrounder.experiments` |

Generic evaluation must not import `benchmarks` or `experiments`. Adapt task records through an
`EvaluationAdapter` before computing neutral metrics.

## Validation Profiles

`PredictionValidator` still detects issues. `ValidationProfile` decides their severity:

- Pipeline runtime: `core`.
- Interactive development and the CLI default: `development`.
- Submission/export: `release`.

Unknown assigned codes remain blocking when a terminology repository is loaded. Offsets, schema,
type/code-system compatibility, duplicate IDs, and relation constraints are blocking in every
profile.

## Test Commands

The default suite excludes slow or environment-sensitive markers:

```bash
uv run pytest tests
```

Use the full public suite before a release:

```bash
uv run pytest -o addopts='' tests
```

Do not remove hard offset, schema, code-system, or relation tests to improve runtime. Move genuine
end-to-end, model, private-data, release-artifact, and benchmark coverage to the matching marker.
