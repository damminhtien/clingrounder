"""Offline contract tests for registered data-source connectors."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import BinaryIO

import pytest

from medical_kg_nlp.mining.connectors import (
    DailyMedConnector,
    LocalArchiveConnector,
    PmcOaConnector,
    StaticHttpConnector,
)
from medical_kg_nlp.mining.records import RedistributionPolicy, SourceRequest
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


def test_dailymed_discovers_a_pinned_filtered_catalog_slice() -> None:
    source = _registry().by_id("dailymed")
    query = (
        "published_date=2026-07-17&published_date_comparison=eq"
        "&pagesize=2&page=1"
    )
    catalog_uri = f"{source.urls[0]}spls.json?{query}"
    payload = json.dumps(
        {
            "data": [
                {
                    "setid": "z-set",
                    "spl_version": 3,
                    "published_date": "Jul 17, 2026",
                    "title": "Drug Z",
                },
                {
                    "setid": "a-set",
                    "spl_version": 1,
                    "published_date": "Jul 17, 2026",
                    "title": "Drug A",
                },
            ],
            "metadata": {
                "current_page": 1,
                "total_pages": 1,
                "total_elements": 2,
                "db_published_date": "Jul 17, 2026 07:50:58PM EST",
            },
        }
    ).encode()
    connector = DailyMedConnector(source, MemoryTransport({catalog_uri: payload}))

    artifacts = list(
        connector.discover(
            SourceRequest(
                source_id="dailymed",
                source_version="catalog-2026-07-17",
                parameters={
                    "catalog": {
                        "expected_db_published_date": "Jul 17, 2026 07:50:58PM EST",
                        "expected_total_elements": 2,
                        "filters": {
                            "published_date": "2026-07-17",
                            "published_date_comparison": "eq",
                        },
                        "page_size": 2,
                    }
                },
            )
        )
    )

    assert [item.metadata["set_id"] for item in artifacts] == ["a-set", "z-set"]
    assert artifacts[0].metadata["spl_version"] == "1"
    assert artifacts[0].metadata["catalog_db_published_date"].startswith("Jul 17")
    assert artifacts[0].metadata["discovery_uri"] == catalog_uri


def test_dailymed_catalog_fails_closed_when_snapshot_changes() -> None:
    source = _registry().by_id("dailymed")
    catalog_uri = f"{source.urls[0]}spls.json?pagesize=100&page=1"
    payload = json.dumps(
        {
            "data": [],
            "metadata": {
                "current_page": 1,
                "total_pages": 0,
                "total_elements": 0,
                "db_published_date": "Jul 18, 2026 01:00:00AM EST",
            },
        }
    ).encode()
    connector = DailyMedConnector(source, MemoryTransport({catalog_uri: payload}))

    with pytest.raises(ValueError, match="DailyMed catalog changed"):
        list(
            connector.discover(
                SourceRequest(
                    source_id="dailymed",
                    source_version="catalog-2026-07-17",
                    parameters={
                        "catalog": {
                            "expected_db_published_date": (
                                "Jul 17, 2026 07:50:58PM EST"
                            )
                        }
                    },
                )
            )
        )


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
                parameters={"pmc_ids": ["PMC42"], "content_format": "oa_package"},
            )
        )
    )

    artifact = connector.fetch(discovered[0], store=LocalArtifactStore(tmp_path))

    assert artifact.license_id == "CC BY 4.0"
    assert artifact.object.sha256 == hashlib.sha256(b"fixture-package").hexdigest()
    assert artifact.metadata["pmc_id"] == "PMC42"
    assert artifact.redistribution is RedistributionPolicy.ATTRIBUTION


def test_pmc_oa_uses_license_resolved_bioc_transport_by_default(tmp_path: Path) -> None:
    source = _registry().by_id("pmc_oa")
    discovery_uri = f"{source.urls[0]}?id=PMC42"
    package_uri = "ftp://ftp.ncbi.example/PMC42.tar.gz"
    bioc_uri = f"{source.urls[1]}/BioC_json/PMC42/unicode"
    transport = MemoryTransport(
        {
            discovery_uri: (
                b'<OA><records><record id="PMC42" license="CC BY-NC-ND">'
                b'<link format="tgz" href="ftp://ftp.ncbi.example/PMC42.tar.gz"/>'
                b"</record></records></OA>"
            ),
            bioc_uri: b'{"documents": [{"id": "42", "passages": []}]}',
        }
    )
    connector = PmcOaConnector(source, transport)

    discovered = list(
        connector.discover(
            SourceRequest(
                source_id="pmc_oa",
                source_version="2026-07-18",
                parameters={"pmc_ids": ["pmc42"]},
            )
        )
    )
    artifact = connector.fetch(discovered[0], store=LocalArtifactStore(tmp_path))

    assert discovered[0].uri == bioc_uri
    assert discovered[0].media_type == "application/json"
    assert artifact.license_id == "CC BY-NC-ND"
    assert artifact.redistribution is RedistributionPolicy.PROHIBITED
    assert artifact.metadata["oa_package_uri"] == package_uri
    assert artifact.metadata["content_format"] == "bioc_json"


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
    assert discovered[0].metadata == {
        "filename": "codiesp.zip",
        "byte_size": str(len(b"local fixture")),
    }
    assert discovered[0].expected_sha256 == hashlib.sha256(b"local fixture").hexdigest()
    assert discovered[0].uri.startswith("file:")
