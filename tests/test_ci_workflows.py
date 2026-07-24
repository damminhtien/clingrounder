"""Contracts that keep GitHub Actions on the repository's locked toolchain."""

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "workflow_path",
    (Path(".github/workflows/ci.yml"), Path(".github/workflows/nightly.yml")),
)
def test_ci_workflows_install_and_run_from_uv_lock(workflow_path: Path) -> None:
    workflow = workflow_path.read_text(encoding="utf-8")

    # INVARIANT: CI must not silently upgrade lint/test tools independently of uv.lock.
    assert 'UV_VERSION: "0.11.31"' in workflow
    assert "uv sync --frozen --extra dev --extra data" in workflow
    assert "pip install -e" not in workflow
    assert "run: uv run" not in workflow
    assert ".venv/bin/ruff check ." in workflow
    assert ".venv/bin/mypy src" in workflow
    assert ".venv/bin/pytest" in workflow
