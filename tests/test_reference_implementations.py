from __future__ import annotations

import json
from pathlib import Path

import pytest

from clingrounder.experiments.reference_implementations import (
    load_reference_registry,
    verify_reference_checkouts,
)


def test_tracked_reference_registry_is_valid() -> None:
    registry = load_reference_registry(
        "configs/references/clinical_nlp_sources.json"
    )

    assert len(registry.sources) == 9
    assert {source.source_id for source in registry.sources} >= {
        "medspacy",
        "negbio",
        "sapbert",
        "vietmed_ner",
        "vihealthbert",
    }
    assert all(len(source.revision) == 40 for source in registry.sources)


def test_registry_rejects_unpinned_revision(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "clinical-nlp-reference-registry.v1",
                "sources": [
                    {
                        "source_id": "bad",
                        "repository_url": "https://example.test/repo.git",
                        "revision": "main",
                        "checkout": "bad",
                        "license_status": "unverified",
                        "license_spdx": None,
                        "license_evidence": [],
                        "inspected_paths": ["README.md"],
                        "adopt": ["architecture"],
                        "reject": ["code copying"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="full SHA-1"):
        load_reference_registry(path)


def test_checkout_verification_reports_missing_paths(tmp_path: Path) -> None:
    registry = load_reference_registry(
        "configs/references/clinical_nlp_sources.json"
    )

    results = verify_reference_checkouts(registry, tmp_path)

    assert not any(result.valid for result in results)
    assert all(result.actual_revision is None for result in results)
