from __future__ import annotations

import subprocess
import sys

from clingrounder import Pipeline
from clingrounder.pipeline import PipelineFactory as ForwardedFactory
from clingrounder.pipeline.advanced import PipelineFactory


def test_pipeline_package_forwards_to_one_advanced_namespace() -> None:
    assert ForwardedFactory is PipelineFactory
    assert Pipeline.__module__ == "clingrounder.pipeline.facade"


def test_root_import_does_not_load_optional_model_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import clingrounder; "
                "assert not any(name in sys.modules for name in "
                "('torch', 'transformers', 'faiss'))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stderr == ""
