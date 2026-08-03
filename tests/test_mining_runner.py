"""Offline end-to-end tests for resumable declarative mining plans."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import yaml

from medical_kg_nlp.cli.main import main
from medical_kg_nlp.mining.runner import (
    artifact_store_from_uri,
    load_mining_plan,
    run_mining_plan,
)


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


def test_cached_acquisition_rehydrates_a_new_artifact_store(tmp_path: Path) -> None:
    plan_path = _fixture_plan(tmp_path)
    first = run_mining_plan(plan_path)
    artifact = json.loads(
        (tmp_path / "work" / "artifacts.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    sha256 = artifact["object"]["sha256"]

    plan_payload = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    plan_payload["artifact_store"]["uri"] = "relocated-artifact-store"
    plan_path.write_text(
        yaml.safe_dump(plan_payload, sort_keys=False),
        encoding="utf-8",
    )

    relocated = run_mining_plan(plan_path)
    resumed = run_mining_plan(plan_path)
    relocated_store = artifact_store_from_uri(
        str(tmp_path / "relocated-artifact-store")
    )

    assert first.cache_misses == 2
    assert relocated.cache_misses == 1
    assert relocated.cache_hits == 1
    assert resumed.cache_misses == 0
    assert resumed.cache_hits == 2
    assert relocated_store.exists(sha256)


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
    assert pilot_artifact["uri"] == artifacts[5]["uri"]
    assert pilot_artifact["metadata"] == artifacts[5]["metadata"]
    assert pilot_artifact["sha256"] == (
        "3c72512e43c1e298c53874bb1d0884dcd8a695c9234890fc769c5054f58bdeb6"
    )


def test_rxnorm_plan_resolves_runtime_paths_but_pins_source_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = tmp_path / "licensed" / "RxNorm_full_07062026.zip"
    monkeypatch.setenv("RXNORM_FULL_ARCHIVE", str(archive))
    monkeypatch.setenv("MEDICAL_KG_ARTIFACT_STORE", str(tmp_path / "external-store"))

    plan = load_mining_plan("configs/mining/rxnorm-full-2026-07-06.yaml")
    source = plan.sources[0]

    assert source.source_id == "rxnorm_full_2026_07_06"
    assert source.parse_documents is False
    assert source.parameters["paths"] == [str(archive.resolve())]
    assert source.parameters["sha256"]["RxNorm_full_07062026.zip"] == (
        "53523ee9f1fcd7ee426698edf566aedebe548a6ec8cc372c41271fc5b28e784c"
    )


def test_phase1_round2_plan_requires_runtime_archive_and_pins_sha256(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = tmp_path / "private" / "input_turn2_vong1.zip"
    monkeypatch.setenv("PHASE1_ROUND2_ARCHIVE", str(archive))
    monkeypatch.setenv(
        "MEDICAL_KG_ARTIFACT_STORE",
        f"file://{tmp_path / 'encrypted-artifact-store'}",
    )

    plan = load_mining_plan("configs/benchmarks/phase1/mining/phase1-round2-2026-07-22.yaml")
    source = plan.sources[0]

    assert source.source_id == "phase1_round2_input"
    assert source.source_version == "round2-phase1-2026-07-22"
    assert source.parameters["paths"] == [str(archive.resolve())]
    assert source.parameters["sha256"]["input_turn2_vong1.zip"] == (
        "989d82404a9c1f3739e15d68a1e69d0f1f90d35c93c04ab0988e071fc1525545"
    )
    assert plan.artifact_store.encrypted_at_rest is True
