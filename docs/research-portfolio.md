# Research Portfolio

ClinGrounder is both a usable clinical NLP toolkit and a record of reproducible engineering
experiments. This page separates what an application can depend on from work that is still being
evaluated.

## Evidence status

| Area | Current status | What the evidence supports |
| --- | --- | --- |
| Core pipeline | Stable alpha API | Local deterministic extraction, context, linking, validation, and raw offsets |
| Vietnamese resource pack | Bundled `vi-clinical-small` | Offline smoke execution and package-data integrity, not full terminology coverage |
| Product benchmark | Synthetic pilot | Runner, metrics, protected gates, and reproducibility contracts |
| Synthetic expansion | 600/100/200 generated documents | Stress and coverage diagnostics only; not clinical generalization |
| Public Vietnamese model | Pending public snapshot | Training contract and governance checks are executable; no model is promoted yet |
| Mining workflows | Research extension | Licensed acquisition, parsing, deduplication, review, and snapshot provenance |
| Competition benchmark | Optional plugin | Historical reproduction and ablation research, not the product quality claim |

The benchmark audit currently reports `eligible_for_clinical_claim: false`. This is intentional:
the checked-in fixture is synthetic and the public model contract remains
`pending_public_snapshot`. No number on this page should be read as clinical validation.

## Stable engineering contributions

- Raw `[start, end)` spans remain owned by the original source text through normalization and
  model projection.
- Candidate linking is constrained by entity type, code system, and active terminology membership.
- Context decisions carry explicit cue scope and provenance rather than opaque booleans.
- Pipeline resources have explicit lifecycle ownership and deterministic close behavior.
- Artifact manifests, versioned caches, and local-only acquisition make resource identity inspectable.
- Neutral evaluation is independent from benchmark-specific schemas and includes protected metrics.

## Experimental tracks

The following modules are replaceable research adapters, not hidden defaults:

- local Hugging Face token classification and cross-encoder adapters;
- dense synonym retrieval and structured RxNorm reranking;
- graph evidence and relation extraction;
- Vietnamese data mining and synthetic challenge generation;
- model training, DAPT, and listwise linking experiments.

The measured Vietnamese NER runs and the rejected DAPT promotion are documented in
[vi-ner-experiments.md](research/vi-ner-experiments.md). Their manifests are provenance evidence
for research reproducibility; they are not public model artifacts.

Each experiment should record its configuration, input fingerprints, model revision, environment,
and output artifact. A local metric can prioritize work, but a release claim requires the relevant
public benchmark and its audit gates.

## Reproduce the current evidence

Run the product pilot and its four deterministic profiles:

```bash
clingrounder-benchmark suite \
  --benchmark benchmarks/vi_clinical_grounding_v1 \
  --config exact=configs/benchmarks/vi_clinical_grounding_v1/exact.yaml \
  --config lexical=configs/benchmarks/vi_clinical_grounding_v1/lexical.yaml \
  --config hybrid=configs/benchmarks/vi_clinical_grounding_v1/hybrid.yaml \
  --config full=configs/benchmarks/vi_clinical_grounding_v1/full.yaml \
  --output artifacts/benchmarks/vi-clinical-grounding-v1/suite
```

Audit the publication gate:

```bash
clingrounder-benchmark audit \
  --benchmark benchmarks/vi_clinical_grounding_v1 \
  --output artifacts/benchmarks/vi-clinical-grounding-v1/audit.json \
  --strict
```

Use [model-training.md](model-training.md) for the public model contract and
[artifacts.md](artifacts.md) for resource identity and offline installation. Use
[evaluation.md](evaluation.md) to interpret exact-span, assertion, linking, relation, and runtime
metrics without conflating them into one opaque score.
