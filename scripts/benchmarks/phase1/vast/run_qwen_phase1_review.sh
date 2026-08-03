#!/usr/bin/env bash
# Run a pinned Qwen missing-entity review against one frozen Phase 1 projection.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../../vast/template_runtime.sh
source "${SCRIPT_DIR}/../../../vast/template_runtime.sh"

REPO_ROOT="${REPO_ROOT:-/workspace/medical-kg}"
TEMPLATE_PYTHON="${TEMPLATE_PYTHON:-/venv/main/bin/python}"
TEMPLATE_BIN="$(dirname "${TEMPLATE_PYTHON}")"
HF_HOME="${HF_HOME:-/workspace/hf}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/pip-cache}"
DOCUMENTS="${DOCUMENTS:-outputs/mining/phase1-round2-hosted-2026-07-27/documents.jsonl}"
REVIEW_SOURCE="${REVIEW_SOURCE:-friend31=outputs/inputs/friend31-strict-known.zip}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/models/phase1-qwen3-friend31-review}"
SOURCE_ARCHIVE_SHA256="${SOURCE_ARCHIVE_SHA256:-989d82404a9c1f3739e15d68a1e69d0f1f90d35c93c04ab0988e071fc1525545}"
CONFIG="${CONFIG:-configs/benchmarks/phase1/models/phase1-qwen3-8b-vietmed-verifier-2026-07-27.yaml}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
MODEL_REVISION="${MODEL_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
REVIEW_MAX_ROUNDS="${REVIEW_MAX_ROUNDS:-2}"
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-28800}"

cd "${REPO_ROOT}"
export HF_HOME PIP_CACHE_DIR

medical_kg_vast_verify_pytorch_template "${TEMPLATE_PYTHON}"
medical_kg_vast_install_project_runtime \
  "${TEMPLATE_PYTHON}" \
  "${REPO_ROOT}" \
  "${PIP_CACHE_DIR}" \
  "accelerate==1.14.0" \
  "pydantic==2.13.4" \
  "PyYAML==6.0.3" \
  "tokenizers==0.22.2" \
  "transformers==5.13.0"

"${TEMPLATE_BIN}/hf" download \
  "${MODEL_ID}" \
  --revision "${MODEL_REVISION}"

# MODEL: the timeout is an operational cost guard. Partial per-document JSON remains available for
# diagnosis, while a clean rerun is deterministic and cannot allow unbounded billing.
timeout --signal=TERM "${MAX_RUNTIME_SECONDS}" \
  "${TEMPLATE_PYTHON}" -m medical_kg_nlp.cli \
  benchmark phase1 qwen propose \
  --config "${CONFIG}" \
  --documents "${DOCUMENTS}" \
  --source-archive-sha256 "${SOURCE_ARCHIVE_SHA256}" \
  --review-source "${REVIEW_SOURCE}" \
  --review-max-rounds "${REVIEW_MAX_ROUNDS}" \
  --review-only \
  --no-adjudication \
  --output-dir "${OUTPUT_DIR}"

test -f "${OUTPUT_DIR}/manifest.json"
