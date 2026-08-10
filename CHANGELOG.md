# Changelog

All notable changes to this project will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions use
pre-release semantic versioning while the public API is still maturing.

## [Unreleased]

### Added

- Automated GitHub release and PyPI publishing workflow using OIDC Trusted Publishing.
- CI package build checks and release documentation.
- Framework-neutral model artifact manifests with SHA-256 and governance verification.
- Public Vietnamese NER training-contract inspection without optional ML dependencies.
- Public benchmark expansion and model-training/release contracts.
- Deterministic synthetic benchmark expansion generator with split/template leakage guards.
- Clean-wheel smoke verification for the bundled offline quickstart in CI and release workflows.
- Task-neutral benchmark dataset audit command and report contract for fingerprint, leakage,
  license, split-policy, and human-review publication gates.
- Refreshed archived GPU and QLoRA run specs after the package lock fingerprint changed in the
  `0.1.0a6` release.
- Strengthened the public Vietnamese model contract: ready runs now require checksummed data
  manifests and an eligible dataset-audit report.
- Added `clingrounder-benchmark suite` to run named product profiles and write a deterministic
  ablation index alongside each complete benchmark artifact bundle.
- Made the neutral dataset benchmark loader derive entity, assertion, and code-system taxonomies
  from each manifest instead of hard-coding the ClinGrounder product task.
- Added a hashed, dataset-bound reviewer-agreement artifact requirement for human-reviewed public
  benchmark releases.
- Added a task-neutral CLI path to export reviewer agreement into that release artifact schema.
- Added a validated review-pack importer that preserves independent submissions, detects exact
  agreement versus disagreement, and keeps gold promotion as an explicit human step.
- Corrected neutral benchmark linking metrics so missed coded entities count as Recall@k/MRR misses
  and assignment coverage reports actual primary-code emission.
- Added an optional offline Streamlit demo for inspecting spans, assertions, candidate provenance,
  and relations without adding UI dependencies to the core package.
- Extended the offline demo with candidate qualification evidence, relation provenance, and
  per-stage runtime latency from the production trace API.
- Preserved dataset, profile, and terminology fingerprints in benchmark summaries and suite
  indexes, with optional provenance enforcement in reference verification.
- Added a CI-regenerated 900-document synthetic diagnostic reference alongside the small public
  contract fixture.
- Added typed artifact manifests, deterministic versioned caches, and explicit local-only artifact
  acquisition with checksum and path-safety validation.
- Documented the measured Vietnamese NER/DAPT research track, including a rejected promotion and
  the provenance boundary that keeps competition checkpoints out of the public runtime.
- Added optional, validated benchmark metrics to artifact manifests and recorded the evidence
  attached to the bundled Vietnamese resource pack.

### Fixed

- Synthetic benchmark templates now select concepts by semantic role, derive assertions from
  explicit cues, and keep normalized document text disjoint across train, validation, and test.
- Synchronized checked-in research run specifications with the current `uv.lock` fingerprint so
  reproducibility validation does not fail on a clean checkout.
- Updated the Vietnamese benchmark reproduction script and pilot table from a current measured
  suite run; runtime values remain explicitly machine-dependent.
- Agreement reports now reject invalid probabilities, inconsistent double-review fractions, and
  values below the manifest's declared publication targets.
- Runtime benchmark output comparisons now fingerprint ZIP member names and contents without
  treating archive timestamps or comments as prediction changes.
- Dataset publication audit now validates annotation spans, raw text ownership, IDs, assertions,
  code-system consistency, and relation endpoints before a release can be eligible.
- CI now runs and uploads the public product benchmark audit and measured ablation suite on every
  push, separate from the historical competition plugin.
- README benchmark latency values and the standalone benchmark results page now reference the
  same current pilot run.

### Changed

- None.

## [0.1.0a9] - 2026-08-09

### Added

- Added a deterministic, gold-blind benchmark review-pack command with coordinator provenance,
  source fingerprints, and independent reviewer assignments.
- Added review-pack documentation and regression tests that prevent benchmark gold from entering
  reviewer payloads.

### Fixed

- Re-pinned checked-in model run specifications to the current environment lock and added a CI
  integrity test so future lock refreshes cannot silently invalidate reproducibility profiles.
- Published measured benchmark provenance for the current synthetic pilot without presenting it as
  clinical validation.

## [0.1.0a8] - 2026-08-09

### Added

- Added structural annotation validation to the public benchmark publication audit, including
  raw-offset ownership, declared taxonomy, code-system, and relation endpoint checks.
- Added a CI job that audits and measures the public product benchmark and uploads its evidence
  artifacts for every push.

### Fixed

- Made benchmark output comparison independent of ZIP archive timestamps and comments by hashing
  member names and contents.

## [0.1.0a7] - 2026-08-09

### Added

- Added a task-neutral benchmark publication audit with split, fingerprint, leakage, license,
  and human-review gates.
- Strengthened the public Vietnamese model training contract with checksummed data manifests and
  an eligible dataset-audit requirement for `ready` runs.

### Fixed

- Refreshed archived GPU and QLoRA run specifications to match the current dependency lock.
- Re-pinned those run specifications after the final `0.1.0a7` lockfile refresh.

## [0.1.0a6] - 2026-08-08

### Fixed

- Refreshed README benchmark measurements and fixed the editable demo installation command.

## [0.1.0a5] - 2026-08-08

### Added

- Expanded the public Vietnamese benchmark generator with laboratory test/result entities and a
  deterministic `HAS_VALUE` relation slice.
- Enabled relation extraction and KG validation in the public `full` benchmark profile.
- Added benchmark relation endpoint/type validation, duplicate-ID checks, and strict public
  entity code-system consistency checks.

### Documentation

- Published measured benchmark evidence for the relation-enabled full profile and documented the
  synthetic benchmark's non-clinical status and remaining human-review gap.

## [0.1.0a4] - 2026-08-08

### Added

- Added a framework-neutral model artifact release contract and inspectable Vietnamese NER
  training contract.
- Added deterministic synthetic benchmark expansion with split/template leakage guards.

### Fixed

- Synchronized research run specifications with the current dependency lock fingerprint.

## [0.1.0a3] - 2026-08-07

### Added

- Published measured product-benchmark metrics, reproducibility checks, and protected metric gates.
- Added a checksum-pinned manifest for the bundled Vietnamese resource pack.
- Added an optional Streamlit inspection demo outside the core runtime.

## [0.1.0a2] - 2026-08-07

### Fixed

- Rebuilt public package metadata and README from the `ClinGrounder` source tree.
- Added the offline Vietnamese quickstart artifact and the independent product benchmark pilot.
- Removed duplicated benchmark section wording from the README.

## [0.1.0a1] - 2026-08-07

### Added

- Offset-safe clinical text grounding with exact raw-text span ownership.
- Typed pipeline contracts for entity extraction, context, terminology retrieval, linking,
  reranking, relations, and validation.
- Rule-based and optional local model adapters for Vietnamese and mixed Vietnamese-English text.
- Full terminology repository interfaces with in-memory and SQLite FTS5 implementations.
- Neutral evaluation, data-mining, provenance, governance, and optional benchmark plugin layers.
- Deterministic CLI entry points for pipeline execution, terminology, evaluation, validation, mining,
  and benchmark workflows.

### Changed

- Renamed the PyPI distribution to `clingrounder`.
- Kept the Python import namespace as `clingrounder` to reflect the package layout.
- Added strict validation for spans, terminology membership, candidates, relations, configuration,
  runtime lifecycle, and release artifacts.

### Documentation

- Added architecture, code-map, reproducibility, public-release, security, and API-stability docs.
- Documented the boundary between reusable toolkit code and optional competition benchmarks.

[Unreleased]: https://github.com/damminhtien/clingrounder/compare/v0.1.0a9...HEAD
[0.1.0a9]: https://github.com/damminhtien/clingrounder/releases/tag/v0.1.0a9
[0.1.0a8]: https://github.com/damminhtien/clingrounder/releases/tag/v0.1.0a8
[0.1.0a7]: https://github.com/damminhtien/clingrounder/releases/tag/v0.1.0a7
[0.1.0a6]: https://github.com/damminhtien/clingrounder/releases/tag/v0.1.0a6
[0.1.0a5]: https://github.com/damminhtien/clingrounder/releases/tag/v0.1.0a5
[0.1.0a4]: https://github.com/damminhtien/clingrounder/releases/tag/v0.1.0a4
[0.1.0a3]: https://github.com/damminhtien/clingrounder/releases/tag/v0.1.0a3
[0.1.0a2]: https://github.com/damminhtien/clingrounder/releases/tag/v0.1.0a2
[0.1.0a1]: https://github.com/damminhtien/clingrounder/releases/tag/v0.1.0a1
