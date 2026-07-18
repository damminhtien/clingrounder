"""Versioned source-specific discovery adapters for supported open APIs."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from typing import Any
from urllib.parse import quote, urlencode

from medical_kg_nlp.mining.connectors.base import RegisteredConnectorAdapter
from medical_kg_nlp.mining.records import DiscoveredArtifact, SourceRequest
from medical_kg_nlp.mining.registry import SourceDefinition

__all__ = [
    "ClinicalTrialsConnector",
    "DailyMedConnector",
    "PmcOaConnector",
    "StaticHttpConnector",
]

_DAILYMED_CATALOG_FILTERS = frozenset(
    {
        "application_number",
        "boxed_warning",
        "dea_schedule_code",
        "doctype",
        "drug_class_code",
        "drug_class_coding_system",
        "drug_name",
        "labeler",
        "manufacturer",
        "marketing_category_code",
        "name_type",
        "ndc",
        "published_date",
        "published_date_comparison",
        "rxcui",
        "setid",
        "unii_code",
    }
)


class StaticHttpConnector(RegisteredConnectorAdapter):
    """Fetch only URLs supplied explicitly by a versioned mining plan."""

    def _discover(self, request: SourceRequest) -> Iterable[DiscoveredArtifact]:
        records = _artifact_records(request.parameters)
        yield from _records_to_artifacts(request, records)


class DailyMedConnector(RegisteredConnectorAdapter):
    """Discover a pinned DailyMed catalog slice or resolve explicit SPL set IDs."""

    connector_revision = "2"

    def _discover(self, request: SourceRequest) -> Iterable[DiscoveredArtifact]:
        records = request.parameters.get("artifacts")
        if records is not None:
            yield from _records_to_artifacts(request, _artifact_records(request.parameters))
            return
        catalog = request.parameters.get("catalog")
        if catalog is not None:
            yield from self._discover_catalog(request, catalog)
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

    def _discover_catalog(
        self,
        request: SourceRequest,
        raw_catalog: Any,
    ) -> Iterable[DiscoveredArtifact]:
        catalog = _mapping(raw_catalog, field_name="catalog")
        allowed_keys = {
            "expected_db_published_date",
            "expected_total_elements",
            "filters",
            "max_records",
            "page_count",
            "page_size",
            "start_page",
        }
        unknown = sorted(set(catalog) - allowed_keys)
        if unknown:
            raise ValueError(f"Unknown DailyMed catalog fields: {', '.join(unknown)}")
        expected_date = _non_empty_string(
            catalog.get("expected_db_published_date"),
            field_name="catalog.expected_db_published_date",
        )
        start_page = _positive_integer(
            catalog.get("start_page", 1), field_name="catalog.start_page"
        )
        page_count = _positive_integer(
            catalog.get("page_count", 1), field_name="catalog.page_count"
        )
        page_size = _positive_integer(
            catalog.get("page_size", 100), field_name="catalog.page_size"
        )
        if page_size > 100:
            raise ValueError("catalog.page_size cannot exceed DailyMed's limit of 100")
        max_records = _optional_positive_integer(
            catalog.get("max_records"), field_name="catalog.max_records"
        )
        expected_total = _optional_non_negative_integer(
            catalog.get("expected_total_elements"),
            field_name="catalog.expected_total_elements",
        )
        filters = _dailymed_filters(catalog.get("filters", {}))
        base_url = _base_url(self.source, "services/v2/")

        discovered: dict[str, DiscoveredArtifact] = {}
        for page in range(start_page, start_page + page_count):
            query = urlencode(
                [*sorted(filters.items()), ("pagesize", page_size), ("page", page)]
            )
            catalog_uri = f"{base_url}spls.json?{query}"
            with closing(self.open_uri(catalog_uri)) as stream:
                payload = json.load(stream)
            rows, metadata = _parse_dailymed_catalog_page(
                payload,
                expected_page=page,
                expected_db_published_date=expected_date,
                expected_total_elements=expected_total,
            )
            for row in rows:
                set_id = _non_empty_string(row.get("setid"), field_name="DailyMed setid")
                artifact = DiscoveredArtifact(
                    source_id=request.source_id,
                    source_version=request.source_version,
                    uri=f"{base_url}spls/{quote(set_id, safe='')}.xml",
                    media_type="application/xml",
                    metadata={
                        "catalog_db_published_date": expected_date,
                        "catalog_page": str(page),
                        "catalog_title": str(row.get("title", "")),
                        "discovery_uri": catalog_uri,
                        "published_date": str(row.get("published_date", "")),
                        "set_id": set_id,
                        "spl_version": str(row.get("spl_version", "")),
                        "total_elements": str(metadata["total_elements"]),
                    },
                )
                previous = discovered.setdefault(set_id, artifact)
                if previous != artifact:
                    raise ValueError(f"Conflicting DailyMed catalog rows for set ID {set_id!r}")
                if max_records is not None and len(discovered) >= max_records:
                    break
            if max_records is not None and len(discovered) >= max_records:
                break
            if page >= int(metadata["total_pages"]):
                break

        # SCALING: stable set-ID order decouples fetch/checkpoint order from catalog paging.
        yield from (discovered[set_id] for set_id in sorted(discovered))


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


def _parse_dailymed_catalog_page(
    payload: Any,
    *,
    expected_page: int,
    expected_db_published_date: str,
    expected_total_elements: int | None,
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    root = _mapping(payload, field_name="DailyMed catalog response")
    metadata = _mapping(root.get("metadata"), field_name="DailyMed catalog metadata")
    db_published_date = _non_empty_string(
        metadata.get("db_published_date"), field_name="DailyMed db_published_date"
    )
    if db_published_date != expected_db_published_date:
        # INVARIANT: a latest-with-snapshot request must fail closed after DailyMed changes.
        raise ValueError(
            "DailyMed catalog changed: expected db_published_date "
            f"{expected_db_published_date!r}, received {db_published_date!r}"
        )
    current_page = _positive_integer(
        metadata.get("current_page"), field_name="DailyMed current_page"
    )
    if current_page != expected_page:
        raise ValueError(
            f"DailyMed returned page {current_page}, expected page {expected_page}"
        )
    total_elements = _optional_non_negative_integer(
        metadata.get("total_elements"), field_name="DailyMed total_elements"
    )
    if total_elements is None:
        raise ValueError("DailyMed total_elements is required")
    if expected_total_elements is not None and total_elements != expected_total_elements:
        raise ValueError(
            "DailyMed catalog size changed: expected "
            f"{expected_total_elements}, received {total_elements}"
        )
    total_pages = _optional_non_negative_integer(
        metadata.get("total_pages"), field_name="DailyMed total_pages"
    )
    if total_pages is None:
        raise ValueError("DailyMed total_pages is required")
    raw_rows = root.get("data")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise ValueError("DailyMed catalog data must be a sequence")
    rows = tuple(_mapping(row, field_name="DailyMed catalog row") for row in raw_rows)
    return rows, {**metadata, "total_elements": total_elements, "total_pages": total_pages}


def _dailymed_filters(value: Any) -> dict[str, str]:
    filters = _mapping(value, field_name="catalog.filters")
    unknown = sorted(set(filters) - _DAILYMED_CATALOG_FILTERS)
    if unknown:
        raise ValueError(f"Unsupported DailyMed catalog filters: {', '.join(unknown)}")
    result: dict[str, str] = {}
    for key, raw in filters.items():
        if not isinstance(raw, (str, int, float, bool)):
            raise ValueError(f"DailyMed filter {key!r} must be a scalar")
        normalized = str(raw).lower() if isinstance(raw, bool) else str(raw).strip()
        if not normalized:
            raise ValueError(f"DailyMed filter {key!r} cannot be empty")
        result[str(key)] = normalized
    return result


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


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    if any(not isinstance(key, str) or not key.strip() for key in value):
        raise ValueError(f"{field_name} keys must be non-empty strings")
    return value


def _non_empty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _positive_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _optional_positive_integer(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, field_name=field_name)


def _optional_non_negative_integer(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _base_url(source: SourceDefinition, expected_suffix: str) -> str:
    if not source.urls:
        raise ValueError(f"Source {source.id!r} has no API URL")
    base_url = source.urls[0]
    if expected_suffix not in base_url:
        raise ValueError(f"Unexpected API root for source {source.id!r}: {base_url}")
    return base_url.rstrip("/") + "/"
