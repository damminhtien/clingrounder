#!/usr/bin/env bash
# Train the learned Phase 1 span/type verifier on a CUDA Vast template.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=template_runtime.sh
source "${SCRIPT_DIR}/template_runtime.sh"

REPO_ROOT="${REPO_ROOT:-/workspace/medical-kg}"
TEMPLATE_PYTHON="${TEMPLATE_PYTHON:-/venv/main/bin/python}"
HF_HOME="${HF_HOME:-/workspace/hf}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/pip-cache}"
DATASET_DIR="${DATASET_DIR:?DATASET_DIR must contain examples.jsonl and manifest.json}"
INITIALIZATION_MODEL="${INITIALIZATION_MODEL:?INITIALIZATION_MODEL must point to the DAPT encoder}"
INITIALIZATION_FINGERPRINT="${INITIALIZATION_FINGERPRINT:?INITIALIZATION_FINGERPRINT is required}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/models/phase1-joint-span-xlmr-dapt-final-fit-2026-08-01}"
MODEL_ID="${MODEL_ID:-FacebookAI/xlm-roberta-base}"
MODEL_REVISION="${MODEL_REVISION:-e73636d4f797dec63c3081bb6ed5c7b0bb3f2089}"
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-10800}"

cd "${REPO_ROOT}"
export HF_HOME PIP_CACHE_DIR

# SCALING: reuse the CUDA image and persistent HF/pip caches. The only large asset copied to a
# fresh worker is the fingerprinted DAPT initializer, never a rebuilt virtual environment.
medical_kg_vast_verify_pytorch_template "${TEMPLATE_PYTHON}"
medical_kg_vast_install_project_runtime \
  "${TEMPLATE_PYTHON}" \
  "${REPO_ROOT}" \
  "${PIP_CACHE_DIR}" \
  "accelerate==1.14.0" \
  "datasets==4.4.2" \
  "pydantic==2.13.4" \
  "PyYAML==6.0.3" \
  "tokenizers==0.22.2" \
  "transformers==5.13.0"

# MODEL: this is a final-fit artifact. It is evaluated only through a strict official submission.
timeout --signal=TERM "${MAX_RUNTIME_SECONDS}" "${TEMPLATE_PYTHON}" -m medical_kg_nlp.cli \
  benchmark phase1 joint-span train \
  --dataset "${DATASET_DIR}/examples.jsonl" \
  --dataset-manifest "${DATASET_DIR}/manifest.json" \
  --output-dir "${OUTPUT_DIR}" \
  --model-id "${MODEL_ID}" \
  --revision "${MODEL_REVISION}" \
  --initialization-model "${INITIALIZATION_MODEL}" \
  --initialization-fingerprint "${INITIALIZATION_FINGERPRINT}" \
  --max-length 384 \
  --train-batch-size 8 \
  --evaluation-batch-size 16 \
  --epochs 5 \
  --learning-rate 0.00002 \
  --weight-decay 0.01 \
  --warmup-ratio 0.08 \
  --seed 42 \
  --bf16

test -f "${OUTPUT_DIR}/run_manifest.json"
