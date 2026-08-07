#!/usr/bin/env bash
# Cross-fit the learned Phase 1 joint span verifier on a cached Vast PyTorch template.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../../vast/template_runtime.sh
source "${SCRIPT_DIR}/../../../vast/template_runtime.sh"

REPO_ROOT="${REPO_ROOT:-/workspace/clingrounder}"
TEMPLATE_PYTHON="${TEMPLATE_PYTHON:-/venv/main/bin/python}"
HF_HOME="${HF_HOME:-/workspace/hf}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/pip-cache}"
DATASET_DIR="${DATASET_DIR:?DATASET_DIR must contain examples.jsonl and manifest.json}"
INITIALIZATION_MODEL="${INITIALIZATION_MODEL:?INITIALIZATION_MODEL must point to the DAPT encoder}"
INITIALIZATION_FINGERPRINT="${INITIALIZATION_FINGERPRINT:?INITIALIZATION_FINGERPRINT is required}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/models/phase1-joint-span-xlmr-dapt-oof-2026-08-01}"
MODEL_ID="${MODEL_ID:-FacebookAI/xlm-roberta-base}"
MODEL_REVISION="${MODEL_REVISION:-e73636d4f797dec63c3081bb6ed5c7b0bb3f2089}"
FOLD_COUNT="${FOLD_COUNT:-5}"
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-21600}"

cd "${REPO_ROOT}"
export HF_HOME PIP_CACHE_DIR

# SCALING: the supplied CUDA template and its persistent cache avoid rebuilding Torch or
# re-downloading a base checkpoint. Only repository code and compact governed artifacts change.
clingrounder_vast_verify_pytorch_template "${TEMPLATE_PYTHON}"
clingrounder_vast_install_project_runtime \
  "${TEMPLATE_PYTHON}" \
  "${REPO_ROOT}" \
  "${PIP_CACHE_DIR}" \
  "accelerate==1.14.0" \
  "datasets==4.4.2" \
  "pydantic==2.13.4" \
  "PyYAML==6.0.3" \
  "tokenizers==0.22.2" \
  "transformers==5.13.0"

# MODEL: one synthetic forward/backward pass validates the template, CUDA runtime, and local
# initializer before all folds consume paid GPU time. It never reads supervised or Round 2 text.
HF_HUB_OFFLINE=1 "${TEMPLATE_PYTHON}" - <<PY
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = "${INITIALIZATION_MODEL}"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    local_files_only=True,
    num_labels=8,
    ignore_mismatched_sizes=True,
).to("cuda")
batch = tokenizer("[GENRE] clinical [ENTITY] đau ngực", return_tensors="pt")
batch = {name: value.to("cuda") for name, value in batch.items()}
loss = model(**batch, labels=torch.tensor([0], device="cuda")).loss
loss.backward()
print({"smoke_loss": round(float(loss.detach().cpu()), 6), "gpu": torch.cuda.get_device_name(0)})
PY

# INVARIANT: OOF observations are candidate identities and probabilities only. Each fold trains
# without its validation documents; calibration may consume this output, final training may not.
timeout --signal=TERM "${MAX_RUNTIME_SECONDS}" "${TEMPLATE_PYTHON}" -m clingrounder.cli \
  benchmark phase1 joint-span train-oof \
  --dataset "${DATASET_DIR}/examples.jsonl" \
  --dataset-manifest "${DATASET_DIR}/manifest.json" \
  --output-dir "${OUTPUT_DIR}" \
  --model-id "${MODEL_ID}" \
  --revision "${MODEL_REVISION}" \
  --initialization-model "${INITIALIZATION_MODEL}" \
  --initialization-fingerprint "${INITIALIZATION_FINGERPRINT}" \
  --fold-count "${FOLD_COUNT}" \
  --inference-device cuda \
  --max-length 384 \
  --train-batch-size 8 \
  --evaluation-batch-size 16 \
  --epochs 5 \
  --learning-rate 0.00002 \
  --weight-decay 0.01 \
  --warmup-ratio 0.08 \
  --seed 42 \
  --bf16

test -f "${OUTPUT_DIR}/manifest.json"
test -f "${OUTPUT_DIR}/oof_observations.jsonl"
