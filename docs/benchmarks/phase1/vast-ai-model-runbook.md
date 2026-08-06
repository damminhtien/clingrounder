# Vast.ai Model Runbook

This runbook is a handoff plan for model experiments after the Rule NER baseline is frozen. The
current deterministic checkpoint and measurements are documented in
[`rule-ner.md`](rule-ner.md); finish reproducing that artifact before renting a GPU. This
runbook does not authorize uploading competition or private clinical text to a third party.

## Local Control Plane

The tested local control plane uses:

```text
vastai 1.0.9 in an isolated uv tool environment on Python 3.11
credential file: ~/.config/vastai/vast_api_key (mode 0600)
SSH private key: ~/.ssh/vast_ed25519 (mode 0600)
Codex skill: ~/.agents/skills/vastai
```

Load the scoped key for automation without copying its literal value into shell configuration:

```bash
export VAST_API_KEY="$(< "$HOME/.config/vastai/vast_api_key")"
vastai show user --raw >/dev/null
```

Register the public key contents, not its path:

```bash
vastai create ssh-key "$(cat "$HOME/.ssh/vast_ed25519.pub")" -y --raw
```

`vastai 1.0.9` was observed to store the path string itself when called as
`vastai create ssh-key ~/.ssh/vast_ed25519.pub`, despite its CLI help describing path input.
Always compare the first two OpenSSH fields returned by `vastai show ssh-keys --raw` with the local
public key before renting a machine.

The installed Codex CLI `0.118.0` does not expose a `codex plugin` subcommand. The official
`vast-ai/vast-cli` `vastai` skill is therefore the active integration. It is high-risk by design
because the CLI can create billable resources and destroy instances, volumes, keys, and endpoints.
Repository-level approval rules in `AGENTS.md` remain authoritative. Restart Codex before expecting
the newly installed skill to appear in a new session.

For account inventory, use `vastai show instances-v1 --all --raw`. On the tested CLI/API pair,
legacy `vastai show instances --raw` returns HTTP 410 instead of an empty list.

## Workloads

The joint Vietnamese XLM-R masked-language plus synonym-contrastive run has a
separate reproducibility guide: [`xlmr-dapt.md`](xlmr-dapt.md).

### A. Five-Type XLM-R Training

The original pinned run is:

```text
configs/benchmarks/phase1/models/phase1-five-type-xlmr-qa-edu-2026-07-26.yaml
```

That three-epoch schedule collapsed to the `O` class in the first executed GPU experiment. The
selected recovery schedule is:

```text
configs/benchmarks/phase1/models/phase1-five-type-xlmr-qa-edu-e12-lr5e5-2026-07-27.yaml
```

See [`experiments/vast-xlmr-2026-07-27.md`](experiments/vast-xlmr-2026-07-27.md) for the executed
template, runtime caveat, ablation metrics, returned fingerprint, calibration thresholds, and
observed cost.

Recommended rental:

| Resource | Requirement |
| --- | --- |
| GPU | One RTX 4090 24 GiB preferred; A5000/A6000/A40/L4/L40S also suitable |
| Minimum GPU | Ampere or newer, compute capability >= 8.0, BF16, >=16 GiB VRAM |
| Host RAM | >=32 GiB |
| CPU | >=8 vCPU |
| Disk | >=100 GiB local SSD; 150 GiB if retaining model cache and checkpoints |
| Network | Stable outbound HTTPS for one pinned Hugging Face download |
| Runtime | Ubuntu 22.04 or newer, NVIDIA driver with CUDA 12-compatible PyTorch |

Use an on-demand/reliable instance for the first run. The checked-in run spec does not silently
resume a partial output directory, so an interruptible instance saves little on this small run.

### Rental Selection Card

Search for one machine, not a multi-GPU host:

```text
GPU count: 1
preferred GPU: RTX 4090 24 GiB
minimum VRAM: 16 GiB
minimum host RAM: 32 GiB
minimum CPU: 8 vCPU
disk allocation: 150 GiB
CUDA capability: 8.0 or newer
network: public SSH and stable outbound HTTPS
availability: on-demand, not interruptible
```

Read-only offer search:

```bash
vastai search offers \
  'gpu_name=RTX_4090 num_gpus=1 verified=true reliability>=0.98 direct_port_count>=1 rentable=true gpu_ram>=24 cpu_ram>=32' \
  --type on-demand \
  --storage 150 \
  --limit 10 \
  -o 'dlperf_usd-' \
  --raw
```

Offer IDs and prices are volatile. Search again immediately before approval; never encode an offer
ID in a script or config.

Prefer a verified host with a strong reliability history and local NVMe storage. Do not pay for
multiple GPUs: the current run has only 135 training chunks and does not implement distributed
training. Reserve one billed hour for bootstrap and the first training run, then extend only after
the runtime check and first evaluation complete. Actual wall time and rental cost must be recorded
in the returned run manifest instead of being estimated as a reproducibility fact.

### B. Qwen Proposal Reproduction

Do not start this workload until all of these fields are recovered and written to a run spec:

```text
model_id
immutable model revision
prompt text and prompt SHA-256
decoding parameters
quantization mode
output schema
```

The old Qwen artifacts are useful proposal evidence but are not reproducible generators. When the
identity is recovered:

| Mode | GPU |
| --- | --- |
| 7B/8B-class 4-bit inference | One 24 GiB GPU |
| BF16 inference or larger batches | One 48 GiB GPU |
| Training/fine-tuning | Not approved for the first experiment |

The recall pass returns only `text` and `type`. It does not return confidence, context anchors, or
offsets. The verifier pass returns only `KEEP`, `DROP`, `REPLACE_BOUNDARY`, or `REPLACE_TYPE` plus
the single replacement field required by that action. Offset projection, repeated-occurrence
recovery, and every replacement are applied locally against the immutable raw document.

Interrupted proposal runs can be continued with the same output directory:

```bash
RESUME=1 OUTPUT_DIR=outputs/models/<existing-run> \
  bash scripts/benchmarks/phase1/vast/run_qwen_phase1_proposals.sh
```

The runner fingerprints the model run spec, document manifest, support sources, and behavior flags
before reusing output. Every reused document JSON is reparsed and checked against the immutable raw
offsets. A changed fingerprint fails closed and requires a new output directory.

#### Evaluate an unmerged QLoRA adapter

The Phase 1 Qwen run spec accepts one optional local PEFT adapter. The adapter is loaded with
`PeftModel.from_pretrained(..., is_trainable=False)` over the pinned base model; inference never
calls `merge_and_unload()`. This keeps base and adapter identities independently auditable.

Add this block to a copy of the pinned Qwen inference YAML:

```yaml
adapter:
  artifact_id: qwen3-phase1-qlora-specialize
  model_id: local/qwen3-phase1-qlora-specialize
  revision: "<40-character training Git commit>"
  parameter_count: <exact adapter state-dict parameter count>
  kind: adapter
  roles: [adjudication, recall, targeted]
  path: outputs/models/<training-run>/final-adapter
  fingerprint: "<SHA-256 of relative paths and file contents>"
  provenance:
    manifest: outputs/models/<training-run>/run_manifest.json
    manifest_sha256: "<SHA-256 of the training manifest>"
```

The base model plus adapter and any auxiliary model must remain at or below nine billion
parameters. Quantization does not reduce this count. Before loading Torch, execution verifies:

- the adapter directory fingerprint;
- `adapter_config.json` and its base-model identity;
- the immutable QLoRA training-manifest hash;
- the training manifest's base model, revision, parameter count, and Git commit.

After PEFT loading, the runtime sums the adapter state-dict tensor sizes and rejects a mismatch
with `parameter_count`. `medical-kg benchmark phase1 qwen inspect --config ...` performs the
artifact/provenance checks without loading model weights. Exact quotes are still projected by the
existing raw-text projector; PEFT output is never trusted to provide offsets.

## Data Boundary

- Remote XLM-R training receives only the reviewed Round 1 train/development model dataset.
- Round 2 may be uploaded only for temporary competition inference because its registry entry now
  uses `authorized_private`; it remains excluded from training, calibration, pseudo-labeling,
  aliases, and runtime memory.
- Never upload DUA, other private, or quarantined text unless its source policy explicitly allows
  processing on Vast.ai.
- If hosted processing is not allowed, train remotely and copy the checkpoint back; run inference
  on an authorized local/private GPU.
- Do not store credentials, API keys, SSH private keys, raw clinical text, or access URLs in Git.
- Destroy the instance and its volume after artifact verification.

## Before Renting

From the local repository:

```bash
git status --short
git rev-parse HEAD

uv run medical-kg benchmark phase1 model-data build \
  --output-dir outputs/mining/model-datasets/phase1-manual-five-type-v1

uv run medical-kg benchmark phase1 model-data augment-regions \
  --output-dir outputs/mining/model-datasets/phase1-manual-five-type-qa-edu-v1

uv run medical-kg model inspect-token-classifier-run \
  --config configs/benchmarks/phase1/models/phase1-five-type-xlmr-qa-edu-2026-07-26.yaml

tar -czf outputs/models/phase1-five-type-xlmr-qa-edu-training-inputs.tar.gz \
  -C outputs/mining/model-datasets/phase1-manual-five-type-qa-edu-v1 \
  spans.jsonl manifest.json build_manifest.json

shasum -a 256 \
  outputs/models/phase1-five-type-xlmr-qa-edu-training-inputs.tar.gz \
  outputs/mining/model-datasets/phase1-manual-five-type-qa-edu-v1/spans.jsonl
```

Expected `spans.jsonl` SHA-256:

```text
093043d9294773d4faffc890c94f346d1c70c5d95181037b6ebf45c94de0d8f4
```

Record the repository commit, `uv.lock` SHA-256, dataset SHA-256, run-spec SHA-256, and archive
SHA-256 in the experiment journal before transfer.

## Vast.ai Handoff

The user should:

1. Rent a machine matching workload A.
2. Select a maintained PyTorch/CUDA image and record its immutable image digest.
3. Attach persistent storage at `/workspace`.
4. Install a temporary public SSH key.
5. Create a local SSH alias named `vast-medical-kg`.
6. Verify `ssh vast-medical-kg nvidia-smi`.

Before transferring any data, the following remote checks must succeed:

```bash
ssh vast-medical-kg 'uname -s && uname -m'
ssh vast-medical-kg 'nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader'
ssh vast-medical-kg 'df -h /workspace'
```

Expected: Linux `x86_64`, one supported GPU with at least 16 GiB VRAM and compute capability 8.0,
and at least 100 GiB free under `/workspace`.

Do not paste passwords or Vast API keys into chat or commit them. Once the SSH alias works, the
coding agent needs only the alias and repository destination.

Transfer the exact Git revision and training archive:

```bash
rsync -av --progress \
  outputs/models/phase1-five-type-xlmr-qa-edu-training-inputs.tar.gz \
  vast-medical-kg:/workspace/
```

Clone the repository on the instance and checkout the recorded commit. If the repository is
private, use a short-lived deploy key or transfer a Git bundle; do not copy a personal token into
shell history.

## Bootstrap And Train

First inspect the template environment:

```bash
/venv/main/bin/python - <<'PY'
import torch

print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
PY
```

For pinned VietMed support inference on `vastai/pytorch`, reuse the template's CUDA-enabled Torch:

```bash
REPO_ROOT=/workspace/medical-kg \
  scripts/benchmarks/phase1/vast/run_vietmed_support.sh \
  2>&1 | tee /workspace/run_vietmed_template.log
```

Do not run `uv sync --extra ml` inside a separate venv for this inference job. That downloads
Torch and all CUDA wheels again even though the template already contains a working CUDA runtime.
The script pins the lightweight Python dependencies, installs the repository editable with
`--no-deps`, and fails early when the template cannot access CUDA.

The same rule applies to Qwen review. Transfer the frozen review ZIP and pinned instruction
manifest, then run:

```bash
REVIEW_SOURCE=friend31=outputs/inputs/friend31-strict-known.zip \
MAX_RUNTIME_SECONDS=28800 \
  scripts/benchmarks/phase1/vast/run_qwen_phase1_review.sh \
  2>&1 | tee /workspace/run_qwen_review.log
```

This mode skips recall, five targeted passes, and adjudication. It performs at most two
missing-only rounds, keeps the frozen source on every overlap, and has an eight-hour process
timeout as a cost guard.

For a full training run that requires the exact lockfile environment rather than the template
Torch, use the isolated lane below and account for its bootstrap time explicitly:

```bash
cd /workspace/ontological-reasoning-in-medical-knowledge-retrieval
git checkout <RECORDED_COMMIT>
git status --porcelain

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv sync --frozen --extra ml

mkdir -p outputs/mining/model-datasets/phase1-manual-five-type-qa-edu-v1
tar -xzf /workspace/phase1-five-type-xlmr-qa-edu-training-inputs.tar.gz \
  -C outputs/mining/model-datasets/phase1-manual-five-type-qa-edu-v1

sha256sum \
  outputs/mining/model-datasets/phase1-manual-five-type-qa-edu-v1/spans.jsonl

uv run hf download FacebookAI/xlm-roberta-base \
  --revision e73636d4f797dec63c3081bb6ed5c7b0bb3f2089

uv run medical-kg model inspect-token-classifier-run \
  --config configs/benchmarks/phase1/models/phase1-five-type-xlmr-qa-edu-2026-07-26.yaml

CUDA_VISIBLE_DEVICES=0 uv run medical-kg model train-token-classifier-run \
  --config configs/benchmarks/phase1/models/phase1-five-type-xlmr-qa-edu-2026-07-26.yaml \
  2>&1 | tee outputs/models/phase1-five-type-xlmr-qa-edu-2026-07-26/train.log
```

The run is valid only if `run_manifest.json` reports the expected model revision, dataset hash,
run-spec hash, lock hash, CUDA device, BF16 runtime, final model fingerprint, and metrics.

The local preflight currently reports:

```text
status: validated_not_executed
records: 155
train/development chunks: 135/20
entities: 3198
dataset SHA-256: 093043d9294773d4faffc890c94f346d1c70c5d95181037b6ebf45c94de0d8f4
```

This is the expected state before renting. Do not rent a machine merely to rediscover a missing
dataset, unpinned model, or invalid run specification.

## Return And Verify

On the instance:

```bash
tar -czf /workspace/phase1-five-type-xlmr-qa-edu-result.tar.gz \
  -C outputs/models/phase1-five-type-xlmr-qa-edu-2026-07-26 \
  final-model run_manifest.json train.log
sha256sum /workspace/phase1-five-type-xlmr-qa-edu-result.tar.gz
```

Copy the archive back, extract it to the same repository-relative output path, then run locally:

```bash
uv run medical-kg model inspect-token-classifier-run \
  --config configs/benchmarks/phase1/models/phase1-five-type-xlmr-qa-edu-2026-07-26.yaml

uv run medical-kg benchmark phase1 model-data calibrate \
  --pipeline-config configs/benchmarks/phase1/pipeline/phase1-five-type-qa-edu-model-only.yaml \
  --output-dir outputs/models/phase1-five-type-qa-edu-calibration
```

Calibration may read only the 16-document development split. It must not open the frozen holdout or
Round 2 labels.

For the selected recovery run, use:

```bash
uv run medical-kg model inspect-token-classifier-run \
  --config configs/benchmarks/phase1/models/phase1-five-type-xlmr-qa-edu-e12-lr5e5-2026-07-27.yaml

uv run medical-kg benchmark phase1 model-data calibrate \
  --pipeline-config configs/benchmarks/phase1/pipeline/phase1-five-type-qa-edu-e12-lr5e5-model-only.yaml \
  --output-dir outputs/models/calibration-e12-lr5e5-2026-07-27
```

Expected final-model fingerprint:

```text
169a02de77d1198e062789271aac563c8000e8aa0ef7e4c95a1ed06ea07d771a
```

## Promotion Gates

Compare three entity sources without assertions or candidates:

```text
rule-only
model-only
hybrid rule + model
```

Promote a model family only when:

- all model spans round-trip to raw text;
- the returned model artifact passes fingerprint verification;
- development WER improves over Rule NER;
- frozen holdout is opened once after thresholds and router policy are fixed;
- holdout WER does not regress;
- schema/offset validation reports zero issues;
- model ID, revision, prompt/config, decoding, dataset, and output hashes are recorded.

The first hybrid policy should keep Rule NER as the base. Add a model-only proposal when it is
non-overlapping and above a per-type development threshold; exact rule/model agreement receives a
separate confidence tier. Do not let a model rewrite an accepted raw span.

## Stop Conditions

Stop the rented instance when any of these occurs:

- runtime gate rejects GPU/BF16 requirements;
- dataset or lock hash differs;
- the pinned checkpoint cannot be fetched;
- training loss is non-finite;
- the saved model fails fingerprint verification;
- policy forbids remote processing of the intended inference data.

Download logs and manifests before destroying the instance. A checkpoint without its manifest and
hashes is not a reusable artifact.
