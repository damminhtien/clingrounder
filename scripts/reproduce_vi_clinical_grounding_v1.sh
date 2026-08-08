#!/usr/bin/env bash
set -euo pipefail

# Reproduce the public synthetic pilot from a source checkout.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT="${1:-artifacts/benchmarks/vi-clinical-grounding-v1/suite}"
exec clingrounder-benchmark suite \
  --benchmark benchmarks/vi_clinical_grounding_v1 \
  --config exact=configs/benchmarks/vi_clinical_grounding_v1/exact.yaml \
  --config lexical=configs/benchmarks/vi_clinical_grounding_v1/lexical.yaml \
  --config hybrid=configs/benchmarks/vi_clinical_grounding_v1/hybrid.yaml \
  --config full=configs/benchmarks/vi_clinical_grounding_v1/full.yaml \
  --output "$OUTPUT"
