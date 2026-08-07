#!/usr/bin/env bash
# Run pinned Qwen recall and type-targeted proposal passes on a Vast PyTorch template.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../../vast/template_runtime.sh
source "${SCRIPT_DIR}/../../../vast/template_runtime.sh"

REPO_ROOT="${REPO_ROOT:-/workspace/clingrounder}"
TEMPLATE_PYTHON="${TEMPLATE_PYTHON:-/venv/main/bin/python}"
TEMPLATE_BIN="$(dirname "${TEMPLATE_PYTHON}")"
HF_HOME="${HF_HOME:-/workspace/hf}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/pip-cache}"
DOCUMENTS="${DOCUMENTS:-outputs/mining/phase1-round2-hosted-2026-07-27/documents.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/models/phase1-qwen3-round2-proposals}"
SOURCE_ARCHIVE_SHA256="${SOURCE_ARCHIVE_SHA256:-989d82404a9c1f3739e15d68a1e69d0f1f90d35c93c04ab0988e071fc1525545}"
CONFIG="${CONFIG:-configs/benchmarks/phase1/models/phase1-qwen3-8b-vietmed-verifier-2026-07-27.yaml}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
MODEL_REVISION="${MODEL_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
SUPPORT_SOURCE="${SUPPORT_SOURCE:-vietmed=outputs/models/phase1-vietmed-ner-round2-support/support}"
RUN_ADJUDICATION="${RUN_ADJUDICATION:-0}"
EXTRACTION_MODE="${EXTRACTION_MODE:-recall_and_targeted}"
RESUME="${RESUME:-0}"
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-21600}"

cd "${REPO_ROOT}"
export HF_HOME PIP_CACHE_DIR

clingrounder_vast_verify_pytorch_template "${TEMPLATE_PYTHON}"
clingrounder_vast_install_project_runtime \
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

command=(
  "${TEMPLATE_PYTHON}" -m clingrounder.cli
  benchmark phase1 qwen propose
  --config "${CONFIG}"
  --documents "${DOCUMENTS}"
  --source-archive-sha256 "${SOURCE_ARCHIVE_SHA256}"
  --support-source "${SUPPORT_SOURCE}"
  --extraction-mode "${EXTRACTION_MODE}"
  --output-dir "${OUTPUT_DIR}"
)
if [[ "${RUN_ADJUDICATION}" != "1" ]]; then
  command+=(--no-adjudication)
fi
if [[ "${RESUME}" == "1" ]]; then
  command+=(--resume)
fi

# MODEL: six extraction passes are intentionally bounded by wall time. Atomic per-document JSON
# files can be resumed after their run fingerprint and raw offsets are revalidated.
timeout --signal=TERM "${MAX_RUNTIME_SECONDS}" "${command[@]}"

test -f "${OUTPUT_DIR}/manifest.json"
