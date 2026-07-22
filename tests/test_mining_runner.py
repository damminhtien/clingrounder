"""Offline end-to-end tests for resumable declarative mining plans."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import yaml

from medical_kg_nlp.cli.main import main
from medical_kg_nlp.mining.runner import load_mining_plan, run_mining_plan


def _write_registry(path: Path) -> None:
    payload = {
        "schema_version": "medical-source-registry.v2",
        "resources": [
            {
                "id": "fixture_codiesp",
                "name": "Fixture CodiEsp",
                "category": "test_corpus",
                "version": "fixture-v1",
                "version_policy": "pinned",
                "access_class": "open",
                "license_id": "CC-BY-4.0",
                "license_url": "https://example.invalid/license",
                "redistribution": "attribution",
                "hosted_processing_allowed": True,
                "retention": "immutable",
                "connector": "local_archive",
                "parser": "codiesp",
                "allowed_uses": ["offline_test"],
            }
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_plan(path: Path) -> None:
    payload = {
        "schema_version": "medical-mining-plan.v1",
        "registry": "registry.yaml",
        "work_dir": "work",
        "artifact_store": {"uri": "artifact-store"},
        "sources": [
            {
                "source_id": "fixture_codiesp",
                "source_version": "fixture-v1",
                "parameters": {
                    "paths": ["fixture.zip"],
                    "media_type": "application/zip",
                },
            }
        ],
        "snapshot": {
            "version": "open-v1",
            "created_at": "2026-07-18T00:00:00+00:00",
            "output_dir": "snapshot",
            "development_fraction": 0.0,
            "write_parquet": False,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _fixture_plan(tmp_path: Path) -> Path:
    _write_registry(tmp_path / "registry.yaml")
    archive_path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("root/train/text_files/1.txt", "Caso clínico uno")
        archive.writestr("root/train/text_files/2.txt", "Caso clínico dos")
    plan_path = tmp_path / "plan.yaml"
    _write_plan(plan_path)
    return plan_path


def test_plan_resolves_relative_paths_and_resumes_completed_stages(
    tmp_path: Path,
) -> None:
    plan_path = _fixture_plan(tmp_path)

    resolved = load_mining_plan(plan_path)
    first = run_mining_plan(plan_path)
    second = run_mining_plan(plan_path)

    assert resolved.registry == str((tmp_path / "registry.yaml").resolve())
    assert first.artifact_count == 1
    assert first.document_count == 2
    assert first.cache_misses == 2
    assert second.cache_hits == 2
    assert second.cache_misses == 0
    assert first.snapshot_id == second.snapshot_id
    assert (tmp_path / "snapshot" / "manifest.json").is_file()
    result = json.loads((tmp_path / "work" / "run_result.json").read_text())
    assert result["document_count"] == 2
    assert result["path_base"] == "mining_plan_directory"
    assert result["work_dir"] == "work"
    assert result["artifact_manifest"] == "work/artifacts.jsonl"
    assert str(tmp_path) not in json.dumps(result)


def test_data_registry_cli_is_installed_and_task_neutral(
    tmp_path: Path, capsys
) -> None:
    plan_path = _fixture_plan(tmp_path)

    exit_code = main(
        [
            "data",
            "registry",
            "validate",
            "--registry",
            str(plan_path.parent / "registry.yaml"),
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["schema_version"] == "medical-source-registry.v2"
    assert output["sources"][0]["id"] == "fixture_codiesp"


def test_full_dailymed_plan_pins_every_human_release_part(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MEDICAL_KG_ARTIFACT_STORE", str(tmp_path / "external-store"))

    plan = load_mining_plan("configs/mining/dailymed-full-human-2026-07-21.yaml")
    artifacts = plan.sources[0].parameters["artifacts"]

    assert len(artifacts) == 17
    assert sum(int(artifact["metadata"]["expected_spl_count"]) for artifact in artifacts) == (
        143_329
    )
    assert all(len(artifact["metadata"]["source_md5"]) == 32 for artifact in artifacts)
    assert plan.snapshot is None

    pilot = load_mining_plan(
        "configs/mining/dailymed-human-rx-part6-2026-07-21.yaml"
    )
    pilot_artifact = pilot.sources[0].parameters["artifacts"][0]
    assert pilot_artifact == artifacts[5]
