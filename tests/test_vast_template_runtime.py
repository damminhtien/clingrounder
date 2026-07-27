"""Contracts for reusing the prebuilt Vast PyTorch runtime."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


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
                'medical_kg_vast_verify_pytorch_template "$2"\n'
                'medical_kg_vast_install_project_runtime "$2" "$3" "$4" '
                '"transformers==5.13.0"'
            ),
            "bash",
            str(helper),
            str(fake_python),
            "/workspace/medical-kg",
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
            "--editable /workspace/medical-kg"
        ),
    ]


def test_vast_runners_source_shared_template_helper() -> None:
    for path in (
        Path("scripts/vast/run_qwen_phase1_review.sh"),
        Path("scripts/vast/run_vietmed_support.sh"),
    ):
        source = path.read_text(encoding="utf-8")
        assert 'source "${SCRIPT_DIR}/template_runtime.sh"' in source
        assert "uv sync" not in source
