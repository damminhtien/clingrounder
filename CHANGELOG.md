# Changelog

All notable changes to this project will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions use
pre-release semantic versioning while the public API is still maturing.

## [Unreleased]

### Added

- Automated GitHub release and PyPI publishing workflow using OIDC Trusted Publishing.
- CI package build checks and release documentation.
- Framework-neutral model artifact manifests with SHA-256 and governance verification.
- Public benchmark expansion and model-training/release contracts.

### Changed

- None.

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

[Unreleased]: https://github.com/damminhtien/clingrounder/compare/v0.1.0a3...HEAD
[0.1.0a3]: https://github.com/damminhtien/clingrounder/releases/tag/v0.1.0a3
[0.1.0a2]: https://github.com/damminhtien/clingrounder/releases/tag/v0.1.0a2
[0.1.0a1]: https://github.com/damminhtien/clingrounder/releases/tag/v0.1.0a1
