#!/usr/bin/env bash
# Materialize two-pass Qwen exact-quote evidence over the governed mixed-genre token bundle.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../../vast/template_runtime.sh
source "${SCRIPT_DIR}/../../../vast/template_runtime.sh"

REPO_ROOT="${REPO_ROOT:-/workspace/medical-kg}"
TEMPLATE_PYTHON="${TEMPLATE_PYTHON:-/venv/main/bin/python}"
TEMPLATE_BIN="$(dirname "${TEMPLATE_PYTHON}")"
HF_HOME="${HF_HOME:-/workspace/hf}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/pip-cache}"
DATASET_DIR="${DATASET_DIR:-outputs/mining/model-datasets/phase1-final-supervision-qa-edu-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/models/phase1-qwen3-final-token-bundle-source-2026-08-01}"
CONFIG="${CONFIG:-configs/benchmarks/phase1/models/phase1-qwen3-8b-qlora-portable-inference-2026-07-31.yaml}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
MODEL_REVISION="${MODEL_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
EXTRACTION_MODE="${EXTRACTION_MODE:-recall_and_targeted}"
RESUME="${RESUME:-0}"
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-21600}"

cd "${REPO_ROOT}"
export HF_HOME PIP_CACHE_DIR

# SCALING: reuse the PyTorch template and shared HF cache. ``hf download`` is a cache hit when the
# template or volume already has this exact revision; it does not create another virtualenv.
medical_kg_vast_verify_pytorch_template "${TEMPLATE_PYTHON}"
medical_kg_vast_install_project_runtime \
  "${TEMPLATE_PYTHON}" \
  "${REPO_ROOT}" \
  "${PIP_CACHE_DIR}" \
  "accelerate==1.14.0" \
  "pydantic==2.13.4" \
  "peft==0.19.1" \
  "PyYAML==6.0.3" \
  "tokenizers==0.22.2" \
  "transformers==5.13.0"

"${TEMPLATE_BIN}/hf" download "${MODEL_ID}" --revision "${MODEL_REVISION}"

command=(
  "${TEMPLATE_PYTHON}" -m medical_kg_nlp.cli
  benchmark phase1 qwen propose-token-bundle
  --config "${CONFIG}"
  --dataset "${DATASET_DIR}/spans.jsonl"
  --dataset-manifest "${DATASET_DIR}/manifest.json"
  --bundle-build-manifest "${DATASET_DIR}/build_manifest.json"
  --output-dir "${OUTPUT_DIR}"
  --extraction-mode "${EXTRACTION_MODE}"
)
if [[ "${RESUME}" == "1" ]]; then
  command+=(--resume)
fi

# MODEL: Qwen emits exact quotes only. The runner persists completed child documents atomically,
# and resume checks raw text hashes before reusing a response.
timeout --signal=TERM "${MAX_RUNTIME_SECONDS}" "${command[@]}"
test -f "${OUTPUT_DIR}/manifest.json"
