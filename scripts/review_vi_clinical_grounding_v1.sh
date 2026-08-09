#!/usr/bin/env bash
set -euo pipefail

# Create a reproducible gold-blind handoff. Reviewer edits stay in the generated artifact
# directory; only fingerprints and the workflow belong in the public repository.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT="${1:-artifacts/review-packs/vi-clinical-grounding-v1}"
exec clingrounder-benchmark review-pack \
  --benchmark benchmarks/vi_clinical_grounding_v1 \
  --split test \
  --reviewer reviewer-1 \
  --reviewer reviewer-2 \
  --double-review-fraction 0.10 \
  --output "$OUTPUT"
