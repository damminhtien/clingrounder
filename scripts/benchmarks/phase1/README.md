# Phase 1 Reproduction Scripts

These thin wrappers preserve historical benchmark commands and artifact workflows. New reusable
functionality belongs in `src/medical_kg_nlp`; prefer `medical-kg benchmark phase1` when command
parity exists.

`vast/` contains pinned benchmark training and inference jobs. Each job reuses the shared
`scripts/vast/template_runtime.sh` helper rather than creating a new environment.

Restricted inputs and generated outputs are not stored here. Resolve them from the benchmark
config and provenance manifest before running a wrapper.
