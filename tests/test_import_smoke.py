from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "medical_kg_nlp.kg.validator",
        "medical_kg_nlp.kg.constraints",
        "medical_kg_nlp.kg.ontology_reasoner",
        "medical_kg_nlp.ner.medication_attribute_extractor",
        "medical_kg_nlp.schema.validator",
        "medical_kg_nlp.ontology.phase1",
        "medical_kg_nlp.evaluation.phase1",
        "medical_kg_nlp.evaluation",
    ],
)
def test_public_modules_import_in_fresh_process(module_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"

    subprocess.run(
        [sys.executable, "-c", f"import importlib; importlib.import_module({module_name!r})"],
        check=True,
        cwd=repo_root,
        env=env,
    )
