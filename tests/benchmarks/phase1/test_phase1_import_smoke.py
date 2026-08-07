"""Import boundary for the optional Phase 1 benchmark plugin."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_phase1_plugin_imports_in_fresh_process() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import clingrounder.benchmarks.phase1.phase1",
        ],
        check=True,
        cwd=repo_root,
        env=env,
    )
