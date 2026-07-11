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
    assert (
        manifest["content_hash"]
        == json.loads(second.manifest_path.read_text(encoding="utf-8"))["content_hash"]
    )
    assert manifest["git_commit"]
    assert manifest["python_version"]
    assert manifest["environment_lock_hash"]
    assert manifest["input_artifacts"][0]["kind"] == "directory"


def test_run_content_hash_changes_when_input_content_changes(tmp_path: Path) -> None:
    input_path = tmp_path / "input.txt"
    input_path.write_text("first", encoding="utf-8")
    first = create_hashed_run_dir(tmp_path / "runs", label="eval", inputs=[input_path])
    first_hash = json.loads(first.manifest_path.read_text(encoding="utf-8"))["content_hash"]

    input_path.write_text("second", encoding="utf-8")
    second = create_hashed_run_dir(tmp_path / "runs", label="eval", inputs=[input_path])
    second_hash = json.loads(second.manifest_path.read_text(encoding="utf-8"))["content_hash"]

    assert first_hash != second_hash


def test_path_in_run_strips_outputs_prefix_for_relative_paths(tmp_path: Path) -> None:
    run_output = create_hashed_run_dir(tmp_path / "runs", label="eval")

    assert path_in_run("outputs/phase1/output", run_output) == run_output.run_dir / "phase1/output"
    assert path_in_run("phase1/output", run_output) == run_output.run_dir / "phase1/output"
