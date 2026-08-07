#!/usr/bin/env bash
set -euo pipefail

# Reproduce the public synthetic pilot from a source checkout.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT="${1:-artifacts/benchmarks/vi-clinical-grounding-v1/full}"
exec clingrounder-benchmark run \
  --benchmark benchmarks/vi_clinical_grounding_v1 \
  --config configs/benchmarks/vi_clinical_grounding_v1/full.yaml \
  --output "$OUTPUT"
