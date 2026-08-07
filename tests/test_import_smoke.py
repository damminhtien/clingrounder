from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_PUBLIC_MODULES = (
    "clingrounder.kg.validator",
    "clingrounder.kg.constraints",
    "clingrounder.kg.ontology_reasoner",
    "clingrounder.ner.medication_attribute_extractor",
    "clingrounder.schema.validator",
    "clingrounder.evaluation",
)


@pytest.mark.integration
def test_public_modules_import_in_one_fresh_process() -> None:
    """Catch import-time coupling without paying for one interpreter per module."""

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    script = (
        "import importlib\n"
        f"modules = {_PUBLIC_MODULES!r}\n"
        "for module_name in modules:\n"
        "    importlib.import_module(module_name)\n"
    )

    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=repo_root,
        env=env,
    )
