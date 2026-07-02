from __future__ import annotations

import json
from pathlib import Path

from medical_kg_nlp.utils.run_output import create_hashed_run_dir, path_in_run


def test_create_hashed_run_dir_is_unique_and_writes_manifest(tmp_path: Path) -> None:
    first = create_hashed_run_dir(tmp_path / "runs", label="phase1", inputs=["data/raw/input"])
    second = create_hashed_run_dir(tmp_path / "runs", label="phase1", inputs=["data/raw/input"])

    assert first.run_dir != second.run_dir
    assert first.run_dir.exists()
    assert second.run_dir.exists()
    assert "_phase1_" in first.run_id

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == first.run_id
    assert manifest["inputs"] == ["data/raw/input"]


def test_path_in_run_strips_outputs_prefix_for_relative_paths(tmp_path: Path) -> None:
    run_output = create_hashed_run_dir(tmp_path / "runs", label="eval")

    assert path_in_run("outputs/phase1/output", run_output) == run_output.run_dir / "phase1/output"
    assert path_in_run("phase1/output", run_output) == run_output.run_dir / "phase1/output"
