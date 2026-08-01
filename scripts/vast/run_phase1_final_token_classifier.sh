#!/usr/bin/env bash
# Build and fit the final XLM-R token source on the existing Vast PyTorch template.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=template_runtime.sh
source "${SCRIPT_DIR}/template_runtime.sh"

REPO_ROOT="${REPO_ROOT:-/workspace/medical-kg}"
TEMPLATE_PYTHON="${TEMPLATE_PYTHON:-/venv/main/bin/python}"
HF_HOME="${HF_HOME:-/workspace/hf}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/pip-cache}"
RUN_CONFIG="${RUN_CONFIG:-configs/models/phase1-five-type-xlmr-final-supervision-2026-08-01.yaml}"
MODEL_ID="${MODEL_ID:-FacebookAI/xlm-roberta-base}"
MODEL_REVISION="${MODEL_REVISION:-e73636d4f797dec63c3081bb6ed5c7b0bb3f2089}"
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-10800}"

if [[ -z "${PHASE1_PART2_ARCHIVE:-}" ]]; then
  printf 'PHASE1_PART2_ARCHIVE must point to the owner-authorized supervised archive.\n' >&2
  exit 2
fi

cd "${REPO_ROOT}"
export HF_HOME PIP_CACHE_DIR

# SCALING: preserve the template's CUDA runtime and populate its local cache once. This avoids
# transferring a large checkpoint through the slower workstation-to-host path.
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

"${TEMPLATE_PYTHON}" - <<PY
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="${MODEL_ID}",
    revision="${MODEL_REVISION}",
)
print({"cached_model": path})
PY

# MODEL: prove that this exact template, checkpoint revision, and CUDA stack can execute a
# token-classification loss before the full final fit consumes the GPU reservation. The test uses
# only a synthetic sentence and never touches either authorized supervision source.
HF_HUB_OFFLINE=1 "${TEMPLATE_PYTHON}" - <<PY
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

model_id = "${MODEL_ID}"
revision = "${MODEL_REVISION}"
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    revision=revision,
    local_files_only=True,
    use_fast=True,
)
model = AutoModelForTokenClassification.from_pretrained(
    model_id,
    revision=revision,
    local_files_only=True,
    num_labels=11,
).to("cuda")
batch = tokenizer("Bệnh nhân khó thở khi gắng sức.", return_tensors="pt")
batch = {name: value.to("cuda") for name, value in batch.items()}
labels = torch.full_like(batch["input_ids"], fill_value=-100)
labels[:, 1] = 0
loss = model(**batch, labels=labels).loss
loss.backward()
print({"smoke_loss": round(float(loss.detach().cpu()), 6), "gpu": torch.cuda.get_device_name(0)})
PY

# INVARIANT: the dataset loader verifies the authorization manifest and LF child-document offsets
# before writing token windows. Round 2 and Friend31 cannot enter this final-fit source.
"${TEMPLATE_PYTHON}" -m medical_kg_nlp.cli \
  benchmark phase1 model-data build-final-fit \
  --output-dir outputs/mining/model-datasets/phase1-final-supervision-five-type-v1

# MODEL: this source feeds the learned joint lattice; it is not independently promoted by local
# metrics. The official BTC artifact remains the only quality decision point.
timeout --signal=TERM "${MAX_RUNTIME_SECONDS}" "${TEMPLATE_PYTHON}" -m medical_kg_nlp.cli \
  model train-token-classifier-run \
  --config "${RUN_CONFIG}"

test -f outputs/models/phase1-five-type-xlmr-final-supervision-2026-08-01/run_manifest.json
