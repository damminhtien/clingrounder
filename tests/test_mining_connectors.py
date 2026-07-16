"""Offline contract tests for registered data-source connectors."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import BinaryIO

import pytest

from medical_kg_nlp.mining.connectors import (
    DailyMedConnector,
    LocalArchiveConnector,
    PmcOaConnector,
    StaticHttpConnector,
)
from medical_kg_nlp.mining.records import SourceRequest
from medical_kg_nlp.mining.registry import load_source_registry
from medical_kg_nlp.mining.storage import LocalArtifactStore


class MemoryTransport:
    """Return isolated streams so fetch and discovery can reopen one fixture URI."""

    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads

    def open(self, uri: str) -> BinaryIO:
        return io.BytesIO(self.payloads[uri])


def _registry():
    return load_source_registry("data/sources/mining_registry.yaml")


def test_dailymed_discovers_versioned_set_ids() -> None:
    source = _registry().by_id("dailymed")
    connector = DailyMedConnector(source, MemoryTransport({}))

    artifacts = list(
        connector.discover(
            SourceRequest(
                source_id="dailymed",
                source_version="2026-07-16",
                parameters={"set_ids": ["z-set", "a-set"]},
            )
        )
    )

    assert [item.metadata["set_id"] for item in artifacts] == ["a-set", "z-set"]
    assert artifacts[0].uri.endswith("/spls/a-set.xml")


def test_pmc_oa_resolves_and_checkpoints_article_license(tmp_path: Path) -> None:
    source = _registry().by_id("pmc_oa")
    endpoint = f"{source.urls[0]}?id=PMC42"
    package_uri = "https://ftp.ncbi.example/PMC42.tar.gz"
    transport = MemoryTransport(
        {
            endpoint: (
                b'<OA><records><record id="PMC42" license="CC BY 4.0">'
                b'<link format="tgz" href="https://ftp.ncbi.example/PMC42.tar.gz"/>'
                b"</record></records></OA>"
            ),
            package_uri: b"fixture-package",
        }
    )
    connector = PmcOaConnector(source, transport)
    discovered = list(
        connector.discover(
            SourceRequest(
                source_id="pmc_oa",
                source_version="2026-07-16",
                parameters={"pmc_ids": ["PMC42"]},
            )
        )
    )

    artifact = connector.fetch(discovered[0], store=LocalArtifactStore(tmp_path))

    assert artifact.license_id == "CC BY 4.0"
    assert artifact.object.sha256 == hashlib.sha256(b"fixture-package").hexdigest()
    assert artifact.metadata["pmc_id"] == "PMC42"


def test_fetch_rejects_checksum_mismatch(tmp_path: Path) -> None:
    source = _registry().by_id("clinicaltrials_v2")
    uri = "https://example.invalid/study.json"
    connector = StaticHttpConnector(source, MemoryTransport({uri: b"actual"}))
    discovered = next(
        iter(
            connector.discover(
                SourceRequest(
                    source_id=source.id,
                    source_version="2026-07-16",
                    parameters={
                        "artifacts": [
                            {
                                "uri": uri,
                                "media_type": "application/json",
                                "sha256": "0" * 64,
                            }
                        ]
                    },
                )
            )
        )
    )

    with pytest.raises(ValueError, match="Checksum mismatch"):
        connector.fetch(discovered, store=LocalArtifactStore(tmp_path))


def test_local_connector_only_imports_explicit_paths(tmp_path: Path) -> None:
    source = _registry().by_id("codiesp")
    archive = tmp_path / "codiesp.zip"
    archive.write_bytes(b"local fixture")
    connector = LocalArchiveConnector(source)

    discovered = list(
        connector.discover(
            SourceRequest(
                source_id="codiesp",
                source_version="zenodo-3837305",
                parameters={"paths": [str(archive)], "media_type": "application/zip"},
            )
        )
    )

    assert len(discovered) == 1
    assert discovered[0].metadata == {"filename": "codiesp.zip"}
    assert discovered[0].uri.startswith("file:")
