#!/usr/bin/env bash
# Materialize one pinned XLM-R five-type model as final-fit joint-span evidence on Vast.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../../vast/template_runtime.sh
source "${SCRIPT_DIR}/../../../vast/template_runtime.sh"

REPO_ROOT="${REPO_ROOT:-/workspace/medical-kg}"
TEMPLATE_PYTHON="${TEMPLATE_PYTHON:-/venv/main/bin/python}"
HF_HOME="${HF_HOME:-/workspace/hf}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/pip-cache}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must point to the final token-classifier directory}"
MODEL_FINGERPRINT="${MODEL_FINGERPRINT:?MODEL_FINGERPRINT is required}"
MODEL_ID="${MODEL_ID:-FacebookAI/xlm-roberta-base}"
BASE_REVISION="${BASE_REVISION:-e73636d4f797dec63c3081bb6ed5c7b0bb3f2089}"
SOURCE_NAME="${SOURCE_NAME:-xlmr}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/models/phase1-xlmr-final-supervision-source-2026-08-01}"
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-7200}"

if [[ -z "${PHASE1_PART2_ARCHIVE:-}" ]]; then
  printf 'PHASE1_PART2_ARCHIVE must point to the owner-authorized supervised archive.\n' >&2
  exit 2
fi

cd "${REPO_ROOT}"
export HF_HOME PIP_CACHE_DIR

# SCALING: preserve the template CUDA runtime and install only application-level dependencies.
# The checkpoint is already local and loading is restricted to its immutable directory.
medical_kg_vast_verify_pytorch_template "${TEMPLATE_PYTHON}"
medical_kg_vast_install_project_runtime \
  "${TEMPLATE_PYTHON}" \
  "${REPO_ROOT}" \
  "${PIP_CACHE_DIR}" \
  "pydantic==2.13.4" \
  "PyYAML==6.0.3" \
  "tokenizers==0.22.2" \
  "transformers==5.13.0"

# MODEL: the source has no outcome policy. It writes only raw span/type proposals plus manifest
# provenance for later learned joint verification.
timeout --signal=TERM "${MAX_RUNTIME_SECONDS}" "${TEMPLATE_PYTHON}" -m medical_kg_nlp.cli \
  benchmark phase1 joint-span materialize-token-source \
  --model-path "${MODEL_PATH}" \
  --model-fingerprint "${MODEL_FINGERPRINT}" \
  --model-id "${MODEL_ID}" \
  --base-revision "${BASE_REVISION}" \
  --output-dir "${OUTPUT_DIR}" \
  --source-name "${SOURCE_NAME}" \
  --device cuda \
  --batch-size 16 \
  --max-length 512 \
  --stride 64

test -f "${OUTPUT_DIR}/manifest.json"
