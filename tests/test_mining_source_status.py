"""Tests for source-processing status and documentation discoverability."""

from __future__ import annotations

import json
from datetime import date

import pytest
from pydantic import ValidationError

from clingrounder.cli.main import main
from clingrounder.mining.curation import load_annotation_curation_policy
from clingrounder.mining.registry import SourceRegistry, load_source_registry
from clingrounder.mining.recognition_knowledge import (
    load_recognition_knowledge_policy,
)
from clingrounder.mining.source_status import (
    SourceProcessingIndex,
    load_source_processing_index,
    validate_source_processing_paths,
)


def _source_registry() -> SourceRegistry:
    return SourceRegistry.model_validate(
        {
            "schema_version": "medical-source-registry.v2",
            "resources": [
                {
                    "id": "source-a",
                    "name": "Source A",
                    "category": "fixture",
                    "version": "1",
                    "version_policy": "pinned",
                    "access_class": "open",
                    "license_id": "CC-BY-4.0",
                    "license_url": "https://example.test/license",
                    "redistribution": "attribution",
                    "hosted_processing_allowed": True,
                    "retention": "immutable",
                    "connector": "fixture",
                    "parser": "fixture",
                    "allowed_uses": ["testing"],
                }
            ],
        }
    )


def test_source_processing_index_validates_registry_docs_and_configs(tmp_path) -> None:
    dossier = tmp_path / "docs" / "source-a.md"
    config = tmp_path / "configs" / "source-a.yaml"
    dossier.parent.mkdir()
    config.parent.mkdir()
    dossier.write_text("# Source A\n", encoding="utf-8")
    config.write_text("source: a\n", encoding="utf-8")
    payload = {
        "schema_version": "medical-source-processing.v1",
        "sources": [
            {
                "source_id": "source-a",
                "state": "promoted",
                "promotion_boundary": "runtime_opt_in",
                "dossier": "docs/source-a.md",
                "verified_on": date(2026, 7, 19),
                "summary": "Fixture source.",
                "run_configs": ["configs/source-a.yaml"],
                "artifact_roots": ["outputs/source-a"],
            }
        ],
    }

    index = SourceProcessingIndex.model_validate(payload)

    assert validate_source_processing_paths(
        index, _source_registry(), repository_root=tmp_path
    ) == ()


def test_processing_index_rejects_runtime_promotion_without_runtime_boundary() -> None:
    with pytest.raises(ValidationError, match="runtime promotion boundary"):
        SourceProcessingIndex.model_validate(
            {
                "schema_version": "medical-source-processing.v1",
                "sources": [
                    {
                        "source_id": "source-a",
                        "state": "promoted",
                        "promotion_boundary": "review_only",
                        "dossier": "docs/source-a.md",
                        "verified_on": "2026-07-19",
                        "summary": "Invalid fixture.",
                    }
                ],
            }
        )


def test_checked_in_processing_index_is_discoverable() -> None:
    index = load_source_processing_index("data/sources/processing_status.yaml")

    assert validate_source_processing_paths(
        index,
        load_source_registry("data/sources/mining_registry.yaml"),
        repository_root=".",
    ) == ()


def test_registry_cli_reports_checked_in_processing_status(capsys) -> None:
    exit_code = main(
        [
            "data",
            "registry",
            "validate",
            "--processing-index",
            "data/sources/processing_status.yaml",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["processing_source_count"] == output["source_count"]
    assert any(item["source_id"] == "pmc_oa" for item in output["processing"])


def test_pmc_recognition_policy_is_train_pinned_and_excludes_lab_results() -> None:
    policy = load_recognition_knowledge_policy(
        "configs/mining/recognition/pmc-rare-cases-ccby-2026-07-19.yaml"
    )

    assert policy.accepted_inventory_sha256 == (
        "4e66b8487ce5bf9615128737b8731a024430eb8d833e32b14e767f0a94fb9878",
    )
    assert policy.mapped_type("DISEASE") is not None
    assert policy.mapped_type("LAB_RESULT") is None


def test_vietbioner_training_views_preserve_broad_source_semantics() -> None:
    curation = load_annotation_curation_policy(
        "configs/mining/curation/vietbioner-ner.yaml"
    )
    recognition = load_recognition_knowledge_policy(
        "configs/mining/knowledge/vietbioner-recognition-v4.yaml"
    )

    assert curation.allowed_entity_types == frozenset({"FINDING", "PROCEDURE"})
    assert curation.allow_discontinuous is True
    assert recognition.accepted_inventory_sha256 == (
        "d8b89bff924fd6b2f8ea189f10c50e6cb7064156cf933b330601897d9154a49b",
    )
    assert recognition.mapped_type("Symptom_and_Disease") is not None
    assert recognition.mapped_type("DateTime") is None
