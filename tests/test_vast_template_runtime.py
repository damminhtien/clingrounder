"""Contracts for reusing the prebuilt Vast PyTorch runtime."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
def test_vast_template_helper_preserves_template_dependencies(tmp_path: Path) -> None:
    """The helper must verify CUDA and install the project without resolving Torch again."""

    calls = tmp_path / "python-calls.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$FAKE_PYTHON_CALLS\"\n"
        "if [[ \"$1\" == \"-\" ]]; then cat >/dev/null; fi\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    helper = Path("scripts/vast/template_runtime.sh").resolve()
    environment = os.environ.copy()
    environment["FAKE_PYTHON_CALLS"] = str(calls)

    subprocess.run(
        [
            "bash",
            "-c",
            (
                'source "$1"\n'
                'clingrounder_vast_verify_pytorch_template "$2"\n'
                'clingrounder_vast_install_project_runtime "$2" "$3" "$4" '
                '"transformers==5.13.0"'
            ),
            "bash",
            str(helper),
            str(fake_python),
            "/workspace/clingrounder",
            "/workspace/pip-cache",
        ],
        check=True,
        env=environment,
    )

    observed = calls.read_text(encoding="utf-8").splitlines()
    assert observed == [
        "-",
        "-m pip install --cache-dir /workspace/pip-cache transformers==5.13.0",
        (
            "-m pip install --cache-dir /workspace/pip-cache --no-deps "
            "--editable /workspace/clingrounder"
        ),
    ]


def test_vast_runners_source_shared_template_helper() -> None:
    for path in (
        Path("scripts/benchmarks/phase1/vast/run_qwen_phase1_proposals.sh"),
        Path("scripts/benchmarks/phase1/vast/run_qwen_phase1_review.sh"),
        Path("scripts/benchmarks/phase1/vast/run_vietmed_support.sh"),
    ):
        source = path.read_text(encoding="utf-8")
        assert 'source "${SCRIPT_DIR}/../../../vast/template_runtime.sh"' in source
        assert "uv sync" not in source


def test_qwen_proposal_runner_defaults_to_bounded_consensus_inference() -> None:
    source = Path("scripts/benchmarks/phase1/vast/run_qwen_phase1_proposals.sh").read_text(
        encoding="utf-8"
    )

    assert 'MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-21600}"' in source
    assert '--support-source "${SUPPORT_SOURCE}"' in source
    assert '--extraction-mode "${EXTRACTION_MODE}"' in source
    assert "command+=(--no-adjudication)" in source
    assert "command+=(--resume)" in source


def test_qwen_final_supervision_runner_installs_declared_adapter_runtime() -> None:
    """A declared local LoRA adapter must not fail after checkpoint download."""

    source = Path("scripts/benchmarks/phase1/vast/run_qwen_final_supervision.sh").read_text(
        encoding="utf-8"
    )

    assert '"peft==0.19.1"' in source
    assert "PHASE1_PART2_ARCHIVE must point" in source
    assert "propose-final-supervision" in source
