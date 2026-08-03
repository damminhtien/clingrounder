# Phase 1 Manual Gold

## Role

This source is a private, locally reviewed benchmark corpus. It contains 100 annotation files tied
to private clinical notes and was used for diagnostic evaluation, error analysis, and supervised
model experiments during the archived Phase 1 work.

## Public Boundary

- Annotation and source-note bytes remain under `data/manual_gold` and local artifact storage.
- Public Git contains no annotation payloads.
- `data/provenance/local-artifacts.json` records path, byte size, SHA-256, policy rule, and source ID.
- `data/sources/mining_registry.yaml` records access, redistribution, retention, and allowed uses.
- Phase 1 import, validation, and evaluation code remains available through the optional benchmark
  plugin.

## Reproduction

An authorized user restores the private source notes and annotations at their documented local
paths, verifies the checksum inventory, then runs the benchmark commands. The public toolkit and
its fast tests do not require this source.

Offsets are always interpreted against the exact source text used during review. Normalized text
must not be substituted when validating or training span labels.
