"""Versioned source-specific discovery adapters for supported open APIs."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from typing import Any
from urllib.parse import quote

from medical_kg_nlp.mining.connectors.base import RegisteredConnectorAdapter
from medical_kg_nlp.mining.records import DiscoveredArtifact, SourceRequest
from medical_kg_nlp.mining.registry import SourceDefinition

__all__ = [
    "ClinicalTrialsConnector",
    "DailyMedConnector",
    "PmcOaConnector",
    "StaticHttpConnector",
]


class StaticHttpConnector(RegisteredConnectorAdapter):
    """Fetch only URLs supplied explicitly by a versioned mining plan."""

    def _discover(self, request: SourceRequest) -> Iterable[DiscoveredArtifact]:
        records = _artifact_records(request.parameters)
        yield from _records_to_artifacts(request, records)


class DailyMedConnector(RegisteredConnectorAdapter):
    """Resolve DailyMed set IDs to SPL XML API resources."""

    def _discover(self, request: SourceRequest) -> Iterable[DiscoveredArtifact]:
        records = request.parameters.get("artifacts")
        if records is not None:
            yield from _records_to_artifacts(request, _artifact_records(request.parameters))
            return
        set_ids = _string_sequence(request.parameters.get("set_ids"), field_name="set_ids")
        base_url = _base_url(self.source, "services/v2/")
        for set_id in sorted(set_ids):
            yield DiscoveredArtifact(
                source_id=request.source_id,
                source_version=request.source_version,
                uri=f"{base_url}spls/{quote(set_id, safe='')}.xml",
                media_type="application/xml",
                metadata={"set_id": set_id},
            )


class ClinicalTrialsConnector(RegisteredConnectorAdapter):
    """Resolve explicit NCT identifiers to ClinicalTrials.gov API v2 JSON."""

    def _discover(self, request: SourceRequest) -> Iterable[DiscoveredArtifact]:
        records = request.parameters.get("artifacts")
        if records is not None:
            yield from _records_to_artifacts(request, _artifact_records(request.parameters))
            return
        nct_ids = _string_sequence(request.parameters.get("nct_ids"), field_name="nct_ids")
        base_url = _base_url(self.source, "api/v2/")
        for nct_id in sorted(nct_ids):
            yield DiscoveredArtifact(
                source_id=request.source_id,
                source_version=request.source_version,
                uri=f"{base_url}studies/{quote(nct_id, safe='')}",
                media_type="application/json",
                metadata={"nct_id": nct_id},
            )


class PmcOaConnector(RegisteredConnectorAdapter):
    """Resolve PMC IDs through the OA service and retain each article license."""

    def _discover(self, request: SourceRequest) -> Iterable[DiscoveredArtifact]:
        records = request.parameters.get("artifacts")
        if records is not None:
            yield from _records_to_artifacts(request, _artifact_records(request.parameters))
            return
        pmc_ids = _string_sequence(request.parameters.get("pmc_ids"), field_name="pmc_ids")
        endpoint = self.source.urls[0].rstrip("?")
        for pmc_id in sorted(pmc_ids):
            uri = f"{endpoint}?id={quote(pmc_id, safe='')}"
            with closing(self.open_uri(uri)) as stream:
                payload = stream.read()
            yield _parse_pmc_oa_response(request, pmc_id, payload)


def _parse_pmc_oa_response(
    request: SourceRequest,
    pmc_id: str,
    payload: bytes,
) -> DiscoveredArtifact:
    root = ET.fromstring(payload)
    record = root.find(".//record")
    if record is None:
        error = root.findtext(".//error", default="unknown PMC OA error")
        raise ValueError(f"PMC OA lookup failed for {pmc_id}: {error}")
    license_id = (record.get("license") or "").strip()
    if not license_id:
        raise ValueError(f"PMC OA record {pmc_id} did not declare a license")
    links = list(record.findall("link"))
    link = next((item for item in links if item.get("format") == "tgz"), None)
    if link is None:
        link = next((item for item in links if item.get("format") in {"xml", "nxml"}), None)
    href = "" if link is None else (link.get("href") or "").strip()
    if not href:
        raise ValueError(f"PMC OA record {pmc_id} has no JATS package link")
    media_type = (
        "application/gzip"
        if link is not None and link.get("format") == "tgz"
        else "application/xml"
    )
    return DiscoveredArtifact(
        source_id=request.source_id,
        source_version=request.source_version,
        uri=href,
        media_type=media_type,
        metadata={"pmc_id": pmc_id, "license_id": license_id},
    )


def _artifact_records(parameters: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    value = parameters.get("artifacts")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("artifacts must be a sequence of mappings")
    if not value:
        raise ValueError("artifacts cannot be empty")
    records: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("each artifact must be a mapping")
        records.append(item)
    return tuple(records)


def _records_to_artifacts(
    request: SourceRequest,
    records: Sequence[Mapping[str, Any]],
) -> Iterable[DiscoveredArtifact]:
    for record in sorted(records, key=lambda item: str(item.get("uri", ""))):
        metadata = record.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("artifact metadata must be a mapping")
        yield DiscoveredArtifact(
            source_id=request.source_id,
            source_version=request.source_version,
            uri=str(record["uri"]),
            media_type=str(record.get("media_type", "application/octet-stream")),
            expected_sha256=(
                None if record.get("sha256") is None else str(record["sha256"])
            ),
            metadata={str(key): str(value) for key, value in metadata.items()},
        )


def _string_sequence(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of strings")
    result = tuple(str(item).strip() for item in value)
    if not result or any(not item for item in result):
        raise ValueError(f"{field_name} must contain non-empty strings")
    return result


def _base_url(source: SourceDefinition, expected_suffix: str) -> str:
    if not source.urls:
        raise ValueError(f"Source {source.id!r} has no API URL")
    base_url = source.urls[0]
    if expected_suffix not in base_url:
        raise ValueError(f"Unexpected API root for source {source.id!r}: {base_url}")
    return base_url.rstrip("/") + "/"
