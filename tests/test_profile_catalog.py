"""Profile discovery and support-status contracts."""

from __future__ import annotations

from pathlib import Path

from medical_kg_nlp.pipeline.profile_catalog import (
    discover_pipeline_profiles,
    inspect_pipeline_profiles,
    validate_pipeline_profile_catalog,
)


def test_checked_in_pipeline_profiles_have_unique_ids_and_valid_metadata() -> None:
    paths = discover_pipeline_profiles("configs/pipeline")
    entries = inspect_pipeline_profiles("configs/pipeline")

    assert paths == tuple(sorted(paths, key=str))
    assert len(entries) == len(paths)
    assert validate_pipeline_profile_catalog(entries) == ()
    assert {entry.profile.profile_id for entry in entries if entry.profile} == {
        "clinical-baseline",
        "full-terminology",
        "full-terminology-kg-exact",
        "general-terminology-vn",
        "mined-vietbioner-silver",
    }


def test_catalog_reports_invalid_profile_without_hiding_other_profiles(tmp_path: Path) -> None:
    root = tmp_path / "profiles"
    root.mkdir()
    (root / "invalid.yaml").write_text(
        "schema_version: medical-kg.pipeline-profile.v1\nprofile: []\n",
        encoding="utf-8",
    )

    entries = inspect_pipeline_profiles(root)

    assert len(entries) == 1
    assert entries[0].error is not None
    assert validate_pipeline_profile_catalog(entries)
