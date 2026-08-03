#!/usr/bin/env bash
# Run pinned VietMed-NER support inference on a Vast PyTorch template.

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
OUTPUT_DIR="${OUTPUT_DIR:-outputs/models/phase1-vietmed-ner-round2-support}"
SOURCE_ARCHIVE_SHA256="${SOURCE_ARCHIVE_SHA256:-989d82404a9c1f3739e15d68a1e69d0f1f90d35c93c04ab0988e071fc1525545}"
CONFIG="${CONFIG:-configs/benchmarks/phase1/models/phase1-vietmed-ner-verifier-2026-07-27.yaml}"
MODEL_ID="${MODEL_ID:-leduckhai/VietMed-NER}"
MODEL_REVISION="${MODEL_REVISION:-cccffb7de14423114f7d4bafc9f736b9d866e446}"
MODEL_INCLUDE="${MODEL_INCLUDE:-xlm-roberta-base-VietMed-NER/*}"

cd "${REPO_ROOT}"
export HF_HOME PIP_CACHE_DIR

medical_kg_vast_verify_pytorch_template "${TEMPLATE_PYTHON}"
medical_kg_vast_install_project_runtime \
  "${TEMPLATE_PYTHON}" \
  "${REPO_ROOT}" \
  "${PIP_CACHE_DIR}" \
  "pydantic==2.13.4" \
  "PyYAML==6.0.3" \
  "tokenizers==0.22.2" \
  "transformers==5.13.0"

"${TEMPLATE_BIN}/hf" download \
  "${MODEL_ID}" \
  --revision "${MODEL_REVISION}" \
  --include "${MODEL_INCLUDE}"

"${TEMPLATE_PYTHON}" -m medical_kg_nlp.cli \
  benchmark phase1 qwen build-vietnamese-support \
  --config "${CONFIG}" \
  --documents "${DOCUMENTS}" \
  --source-archive-sha256 "${SOURCE_ARCHIVE_SHA256}" \
  --output-dir "${OUTPUT_DIR}"

test -f "${OUTPUT_DIR}/manifest.json"
