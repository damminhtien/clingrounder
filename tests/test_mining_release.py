"""Portable-lock tests for mining datasets, knowledge, and model inputs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from medical_kg_nlp.cli.main import main
from medical_kg_nlp.mining.release import (
    MiningReleaseLock,
    build_mining_release_lock,
    load_mining_release_spec,
    verify_mining_release_lock,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_release_lock_is_deterministic_and_has_no_machine_paths(tmp_path: Path) -> None:
    root, spec_path = _release_fixture(tmp_path / "origin")
    lock_path = root / "release.lock.json"

    first = build_mining_release_lock(spec_path, lock_path)
    first_bytes = lock_path.read_bytes()
    second = build_mining_release_lock(spec_path, lock_path)

    assert first == second
    assert lock_path.read_bytes() == first_bytes
    assert first["artifacts"][0]["path"] == "data/documents.jsonl"
    assert str(root) not in first_bytes.decode("utf-8")
    assert verify_mining_release_lock(lock_path, release_root=root)["valid"] is True


def test_release_lock_verifies_after_tree_moves_to_another_machine_root(
    tmp_path: Path,
) -> None:
    root, spec_path = _release_fixture(tmp_path / "origin")
    lock_path = root / "release.lock.json"
    build_mining_release_lock(spec_path, lock_path)
    relocated = tmp_path / "relocated-checkout"
    shutil.copytree(root, relocated)

    report = verify_mining_release_lock(
        relocated / "release.lock.json",
        release_root=relocated,
    )

    assert report["valid"] is True
    assert report["verified_artifact_count"] == 2
    assert report["errors"] == []


def test_release_verification_detects_content_changes_and_missing_optional(
    tmp_path: Path,
) -> None:
    root, spec_path = _release_fixture(tmp_path / "origin", optional=True)
    lock_path = root / "release.lock.json"
    build_mining_release_lock(spec_path, lock_path)
    (root / "data" / "documents.jsonl").write_text("changed\n", encoding="utf-8")

    report = verify_mining_release_lock(lock_path, release_root=root)
    strict_report = verify_mining_release_lock(
        lock_path,
        release_root=root,
        require_optional=True,
    )

    assert report["valid"] is False
    assert "sha256_mismatch:documents" in report["errors"]
    assert report["optional_missing_artifact_ids"] == ["model-checkpoint"]
    assert "missing_artifact:model-checkpoint" in strict_report["errors"]


def test_release_directory_excludes_are_explicit_and_portable(tmp_path: Path) -> None:
    root, spec_path = _release_fixture(tmp_path / "origin", code_cache=True)
    lock_path = root / "release.lock.json"
    first = build_mining_release_lock(spec_path, lock_path)
    cache_file = root / "src" / "package" / "__pycache__" / "module.pyc"
    cache_file.write_bytes(b"different interpreter cache")
    second = build_mining_release_lock(spec_path, lock_path)

    first_code = next(row for row in first["artifacts"] if row["id"] == "code")
    second_code = next(row for row in second["artifacts"] if row["id"] == "code")
    assert first_code["exclude"] == ["__pycache__", "**/__pycache__", "*.pyc", "**/*.pyc"]
    assert first_code["sha256"] == second_code["sha256"]


def test_release_spec_rejects_absolute_artifact_paths(tmp_path: Path) -> None:
    root, spec_path = _release_fixture(tmp_path / "origin")
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    payload["artifacts"][0]["path"] = str(root / "data" / "documents.jsonl")
    spec_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact paths must be relative"):
        load_mining_release_spec(spec_path)


def test_release_spec_cannot_exclude_scientific_data(tmp_path: Path) -> None:
    _, spec_path = _release_fixture(tmp_path / "origin")
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    payload["artifacts"][0]["exclude"] = ["*.jsonl"]
    spec_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="allowed only for implementation artifacts"):
        load_mining_release_spec(spec_path)


def test_release_cli_lock_and_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root, spec_path = _release_fixture(tmp_path / "origin")
    lock_path = root / "release.lock.json"

    assert main(["data", "release", "lock", "--spec", str(spec_path), "--output", str(lock_path)]) == 0
    lock_output = json.loads(capsys.readouterr().out)
    assert lock_output["artifact_count"] == 2

    assert main(
        [
            "data",
            "release",
            "verify",
            "--manifest",
            str(lock_path),
            "--root",
            str(root),
        ]
    ) == 0
    verify_output = json.loads(capsys.readouterr().out)
    assert verify_output["valid"] is True


def test_checked_in_open_release_contract_is_portable_and_strict() -> None:
    loaded = load_mining_release_spec(
        _REPOSITORY_ROOT
        / "configs/mining/releases/open-ner-retrieval-v1.yaml"
    )
    lock_path = _REPOSITORY_ROOT / "data/releases/open-ner-retrieval-v1.lock.json"
    lock_text = lock_path.read_text(encoding="utf-8")
    lock = MiningReleaseLock.model_validate(json.loads(lock_text))

    assert loaded.spec.release_id == lock.release_id == "open-ner-retrieval-v1"
    assert len(lock.artifacts) == 30
    assert "/Users/" not in lock_text
    assert "/home/" not in lock_text


def _release_fixture(
    base: Path,
    *,
    optional: bool = False,
    code_cache: bool = False,
) -> tuple[Path, Path]:
    root = base / "repository"
    config_dir = root / "configs"
    data_dir = root / "data"
    code_dir = root / "src" / "package"
    config_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    code_dir.mkdir(parents=True)
    (data_dir / "documents.jsonl").write_text('{"text":"Tăng huyết áp"}\n', encoding="utf-8")
    (code_dir / "module.py").write_text('VERSION = "v1"\n', encoding="utf-8")
    if code_cache:
        cache_dir = code_dir / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "module.pyc").write_bytes(b"machine-specific cache")
    artifacts: list[dict[str, object]] = [
        {
            "id": "documents",
            "role": "model_dataset",
            "path": "data/documents.jsonl",
            "description": "Immutable raw-offset fixture documents.",
        },
        {
            "id": "code",
            "role": "implementation",
            "path": "src/package",
            "description": "Code that materialized the dataset.",
            "exclude": ["__pycache__", "**/__pycache__", "*.pyc", "**/*.pyc"],
        },
    ]
    if optional:
        artifacts.append(
            {
                "id": "model-checkpoint",
                "role": "model_checkpoint",
                "path": "models/checkpoint",
                "description": "Optional local GPU-trained checkpoint.",
                "required": False,
            }
        )
    spec = {
        "schema_version": "medical-mining-release-spec.v1",
        "release_id": "fixture-ner-retrieval-v1",
        "description": "Portable fixture release.",
        "release_root": "..",
        "artifacts": artifacts,
        "rebuild_steps": [
            {
                "id": "build-dataset",
                "description": "Rebuild the immutable dataset.",
                "command": "uv run medical-kg data dataset build --output data/documents.jsonl",
            }
        ],
    }
    spec_path = config_dir / "release.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return root, spec_path
