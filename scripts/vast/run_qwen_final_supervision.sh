#!/usr/bin/env bash
# Materialize Qwen exact-quote evidence for the governed Phase 1 final-fit corpus on Vast.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=template_runtime.sh
source "${SCRIPT_DIR}/template_runtime.sh"

REPO_ROOT="${REPO_ROOT:-/workspace/medical-kg}"
TEMPLATE_PYTHON="${TEMPLATE_PYTHON:-/venv/main/bin/python}"
TEMPLATE_BIN="$(dirname "${TEMPLATE_PYTHON}")"
HF_HOME="${HF_HOME:-/workspace/hf}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/pip-cache}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/models/phase1-qwen3-final-supervision-source-2026-07-31}"
CONFIG="${CONFIG:-configs/benchmarks/phase1/models/phase1-qwen3-8b-qlora-portable-inference-2026-07-31.yaml}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
MODEL_REVISION="${MODEL_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
EXTRACTION_MODE="${EXTRACTION_MODE:-recall_and_targeted}"
RESUME="${RESUME:-0}"
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-21600}"

if [[ -z "${PHASE1_PART2_ARCHIVE:-}" ]]; then
  printf 'PHASE1_PART2_ARCHIVE must point to the owner-authorized supervised archive.\n' >&2
  exit 2
fi

cd "${REPO_ROOT}"
export HF_HOME PIP_CACHE_DIR

# SCALING: use the CUDA-enabled PyTorch template and shared HF/pip caches. Do not create a new
# virtual environment or redownload a cached checkpoint on every resumable proposal run.
# MODEL: this pinned inference config declares a local LoRA adapter, so PEFT is a required runtime
# dependency rather than an optional training-only extra.
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
  benchmark phase1 qwen propose-final-supervision
  --config "${CONFIG}"
  --output-dir "${OUTPUT_DIR}"
  --extraction-mode "${EXTRACTION_MODE}"
)
if [[ "${RESUME}" == "1" ]]; then
  command+=(--resume)
fi

# MODEL: per-document JSON is atomically persisted and can resume only after run-spec and raw
# text hashes match, limiting cost while preserving an immutable source artifact.
timeout --signal=TERM "${MAX_RUNTIME_SECONDS}" "${command[@]}"
test -f "${OUTPUT_DIR}/manifest.json"
