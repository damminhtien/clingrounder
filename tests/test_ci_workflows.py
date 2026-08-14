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


def test_ci_runs_and_uploads_the_public_product_benchmark() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "name: Public product benchmark" in workflow
    assert ".venv/bin/clingrounder-benchmark audit" in workflow
    assert ".venv/bin/clingrounder-benchmark suite" in workflow
    assert ".venv/bin/clingrounder-benchmark verify-reference" in workflow
    assert "scripts/generate_vi_clinical_benchmark.py" in workflow
    assert "scripts/review_vi_clinical_synthetic.py" in workflow
    assert "technical-review.json" in workflow
    assert "synthetic_diagnostic_expected_results.yaml" in workflow
    assert ".venv/bin/clingrounder-benchmark review-pack" in workflow
    assert "--double-review-fraction 1.0" in workflow
    assert "public-benchmark-${{ github.sha }}" in workflow
