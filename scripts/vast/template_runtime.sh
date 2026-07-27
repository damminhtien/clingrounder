#!/usr/bin/env bash
# Shared setup helpers for jobs that reuse a Vast PyTorch template.

# This file is sourced by task-specific runners. Do not enable shell options here because the
# caller owns its execution policy.

medical_kg_vast_verify_pytorch_template() {
  local template_python="$1"

  if [[ ! -x "${template_python}" ]]; then
    printf 'Vast template Python is unavailable: %s\n' "${template_python}" >&2
    return 2
  fi

  # SCALING: the template owns Torch and CUDA. Fail before dependency installation when the
  # selected image cannot expose its GPU to the container.
  "${template_python}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("The Vast PyTorch template cannot access CUDA")
print(
    {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
)
PY
}

medical_kg_vast_install_project_runtime() {
  local template_python="$1"
  local repo_root="$2"
  local pip_cache_dir="$3"
  shift 3

  # MODEL: task libraries are pinned by each runner. The editable project deliberately uses
  # `--no-deps` so pip cannot replace the template's CUDA-enabled Torch distribution.
  "${template_python}" -m pip install \
    --cache-dir "${pip_cache_dir}" \
    "$@"
  "${template_python}" -m pip install \
    --cache-dir "${pip_cache_dir}" \
    --no-deps \
    --editable "${repo_root}"
}
