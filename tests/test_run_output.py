from __future__ import annotations

import json
from pathlib import Path

from clingrounder.utils.run_output import (
    collect_git_metadata,
    create_hashed_run_dir,
    path_in_run,
)


def test_create_hashed_run_dir_is_unique_and_writes_manifest(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "note.txt").write_text("clinical note", encoding="utf-8")

    first = create_hashed_run_dir(tmp_path / "runs", label="phase1", inputs=[input_dir])
    second = create_hashed_run_dir(tmp_path / "runs", label="phase1", inputs=[input_dir])

    assert first.run_dir != second.run_dir
    assert first.run_dir.exists()
    assert second.run_dir.exists()
    assert "_phase1_" in first.run_id

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == first.run_id
    assert manifest["inputs"] == [str(input_dir)]
    assert (
        manifest["content_hash"]
        == json.loads(second.manifest_path.read_text(encoding="utf-8"))["content_hash"]
    )
    assert manifest["git_commit"]
    assert manifest["python_version"]
    assert manifest["environment_lock_hash"]
    assert manifest["input_artifacts"][0]["kind"] == "directory"
    assert manifest["input_artifacts"][0]["file_count"] == 1


def test_run_content_hash_changes_when_input_content_changes(tmp_path: Path) -> None:
    input_path = tmp_path / "input.txt"
    input_path.write_text("first", encoding="utf-8")
    first = create_hashed_run_dir(tmp_path / "runs", label="eval", inputs=[input_path])
    first_hash = json.loads(first.manifest_path.read_text(encoding="utf-8"))["content_hash"]

    input_path.write_text("second", encoding="utf-8")
    second = create_hashed_run_dir(tmp_path / "runs", label="eval", inputs=[input_path])
    second_hash = json.loads(second.manifest_path.read_text(encoding="utf-8"))["content_hash"]

    assert first_hash != second_hash


def test_collect_git_metadata_falls_back_to_source_commit_marker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    marker = tmp_path / ".source-commit"
    marker.write_text("a" * 40 + "\n", encoding="ascii")
    monkeypatch.chdir(tmp_path)

    metadata = collect_git_metadata()

    assert metadata["git_commit"] == "a" * 40
    assert metadata["git_dirty"] is None
    assert metadata["source_control_mode"] == "source_commit_marker"
    assert metadata["source_commit_marker_sha256"] is not None


def test_path_in_run_strips_outputs_prefix_for_relative_paths(tmp_path: Path) -> None:
    run_output = create_hashed_run_dir(tmp_path / "runs", label="eval")

    assert path_in_run("outputs/phase1/output", run_output) == run_output.run_dir / "phase1/output"
    assert path_in_run("phase1/output", run_output) == run_output.run_dir / "phase1/output"
