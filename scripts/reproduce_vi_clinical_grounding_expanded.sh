#!/usr/bin/env bash
set -euo pipefail

# Reproduce the larger, redistributable synthetic diagnostic snapshot.  The snapshot is useful
# for coverage and pipeline-drift checks; it is deliberately not presented as clinical evidence.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT="${1:-artifacts/benchmarks/vi-clinical-grounding-synthetic-v1}"
DATASET="$OUTPUT/dataset"
SUITE="$OUTPUT/suite"
if [[ -n "${CLINGROUNDER_BIN:-}" ]]; then
  CLI_CMD=("$CLINGROUNDER_BIN")
else
  CLI_CMD=(uv run clingrounder-benchmark)
fi
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("$PYTHON_BIN")
else
  PYTHON_CMD=(uv run python)
fi

mkdir -p "$OUTPUT"

# INVARIANT: generation, audit, scoring, and reference verification all use the same dataset
# directory and therefore the same recorded split fingerprints.
"${PYTHON_CMD[@]}" scripts/generate_vi_clinical_benchmark.py \
  --output-dir "$DATASET" \
  > "$OUTPUT/generation-manifest.json"

"${CLI_CMD[@]}" audit \
  --benchmark "$DATASET" \
  --output "$OUTPUT/audit.json"

"${CLI_CMD[@]}" suite \
  --benchmark "$DATASET" \
  --config exact=configs/benchmarks/vi_clinical_grounding_v1/exact.yaml \
  --config lexical=configs/benchmarks/vi_clinical_grounding_v1/lexical.yaml \
  --config hybrid=configs/benchmarks/vi_clinical_grounding_v1/hybrid.yaml \
  --config full=configs/benchmarks/vi_clinical_grounding_v1/full.yaml \
  --output "$SUITE"

"${CLI_CMD[@]}" verify-reference \
  --suite "$SUITE/suite.json" \
  --reference benchmarks/vi_clinical_grounding_v1/synthetic_diagnostic_expected_results.yaml \
  --output "$OUTPUT/reference-verification.json"

printf 'Expanded synthetic benchmark artifacts written to %s\n' "$OUTPUT"
