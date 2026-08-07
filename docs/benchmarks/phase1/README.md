# Archived Phase 1 Benchmark

This directory preserves a Vietnamese medical extraction competition as an optional benchmark
plugin. It records task schema, annotation conventions, experiment designs, official-result
decisions, and reproduction metadata without defining reusable toolkit defaults.

## Ownership

```text
src/clingrounder/benchmarks/phase1/   task code, adapters, export, scoring
configs/benchmarks/phase1/              pipeline, model, submission, experiment specs
tests/benchmarks/phase1/                opt-in public benchmark tests
docs/benchmarks/phase1/                 historical analysis and decisions
```

Core pipeline, context, retrieval, linking, and evaluation packages cannot import this plugin.
Architecture tests enforce that boundary.

## Run

```bash
uv run clingrounder-benchmark list
uv run clingrounder-benchmark phase1 --help
uv run pytest -o addopts='' -m "benchmark and not private and not model" \
  tests/benchmarks/phase1
```

The toolkit quickstart does not load benchmark resources, heuristics, reviewed memory, or output
artifacts.

## Data Boundary

The public repository keeps source policies, fingerprints, configs, and processing dossiers.
Restricted documents, manual annotations, licensed terminology bytes, checkpoints, and generated
predictions remain outside Git. Their local identities are inventoried in
`data/provenance/local-artifacts.json`.

Relevant dossiers remain discoverable under `docs/mining-sources/` because they describe data
lineage rather than runtime ownership.

## Historical Material

- [Evaluation and campaign history](evaluation-history.md)
- [Dictionary and terminology campaign history](dictionary-history.md)
- [Competition engineering playbook](competition-playbook.md)
- [Max-score pipeline](max-score-pipeline.md)
- [Under-9B inference budget](under9b-inference-budget.md)
- [Vast.ai model runbook](vast-ai-model-runbook.md)
- [Vietnamese XLM-R DAPT runbook](xlmr-dapt.md)
- [Experiment records](experiments/)
- [Review audits](reviews/)
- [Source-of-truth notes](source-of-truths/)
- [Early linking plan](linking-plan.md)

Historical documents may discuss rejected models or private artifacts. They are research records,
not recommended defaults for new applications.
