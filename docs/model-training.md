# Model Training And Release Contract

ClinGrounder keeps model training optional. The deterministic local pipeline is the product
baseline; a model is publishable only when its inputs, revision, artifact digest, and evaluation
are inspectable without importing a model framework.

## Training flow

1. Register every dataset snapshot with a source fingerprint, license/access class, and split
   policy. Do not use private or restricted text in a public artifact.
2. Pin the base model by immutable revision and record the model license and intended use.
3. Run the task-specific training contract. The existing token-classifier and DAPT run specs are
   research adapters, not hidden defaults.
4. Evaluate on a source-held-out, template-held-out dataset with fixed seed and environment.
5. Write a `clingrounder.model-artifact.v1` manifest next to the model directory and verify its
   SHA-256 before inference or publication.

The manifest records:

- model ID and immutable revision;
- model artifact and training-config SHA-256 values;
- dataset snapshot fingerprints;
- task and metric names;
- intended/excluded use and known limitations;
- approval state and optional rollback model.

The reusable verifier is available as:

```python
from clingrounder.governance import load_model_artifact_manifest, verify_model_artifact

manifest = load_model_artifact_manifest("model/manifest.json")
verify_model_artifact("model", manifest, require_approved=True)
```

This check is framework-neutral and can run in CI or on a deployment host before loading
Transformers, PyTorch, or another optional runtime.

## Promotion policy

The public product benchmark is currently a **synthetic pilot**. Its perfect pilot scores are
smoke-test evidence, not clinical validation. No Vietnamese model is shipped in the wheel until a
redistributable, human-reviewed benchmark snapshot and a model artifact manifest exist.

For a future model release, compare it against the deterministic baseline and report exact span
F1, type accuracy, assertion metrics, linking recall, offset validity, initialization time, and
peak RSS. A model is not promoted when it improves one aggregate metric by regressing offset
validity, terminology membership, protected slices, or reproducibility.

## Reproducing a research run

The training extension is installed separately:

```bash
uv sync --extra dev --extra ml
uv run clingrounder-research model inspect-token-classifier-run \
  --config configs/benchmarks/phase1/models/<run-spec>.yaml
```

The product-facing Vietnamese NER contract can be inspected without ML dependencies:

```bash
clingrounder-research model inspect-public-training-contract \
  --config configs/training/vi_clinical_ner_v1.yaml
```

The checked-in contract currently reports `pending_public_snapshot`. This is deliberate: it is a
reproducible training specification, not a claim that a public human-reviewed training corpus or
released Vietnamese model already exists.

Competition-specific configs and restricted inputs remain optional benchmark/research material.
They are not part of the quickstart and must not be described as a public ClinGrounder model.

## Limitations

- No hosted model API is required or used by the core package.
- Model quality depends on the licensed, pinned data snapshot and genre distribution.
- A manifest proves identity and provenance, not clinical safety or regulatory approval.
- Model files remain external artifacts unless their license, size, and redistribution policy are
  explicitly satisfied.
