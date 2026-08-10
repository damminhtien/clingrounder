"""Keep public README claims aligned with the declared stable product scope."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readmes_do_not_promote_experimental_assertions_to_stable() -> None:
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    vietnamese = (ROOT / "README_VI.md").read_text(encoding="utf-8")

    assert "in the stable v1\n  surface" in english
    assert "remain experimental" in english
    assert "vẫn là experimental" in vietnamese

    # The old broad claim silently presented all assertion enum values as v1 behavior.
    assert "present, negated, historical, family, possible, planned, conditional, and resolved" not in english


def test_public_docs_use_the_current_repository_path() -> None:
    runbook = (
        ROOT / "docs" / "benchmarks" / "phase1" / "vast-ai-model-runbook.md"
    ).read_text(encoding="utf-8")

    assert "/workspace/ontological-reasoning-in-medical-knowledge-retrieval" not in runbook
    assert "cd /workspace/clingrounder" in runbook
