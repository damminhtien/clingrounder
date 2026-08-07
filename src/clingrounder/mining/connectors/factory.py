"""Composition helpers for source connectors declared in registry v2."""

from __future__ import annotations

from clingrounder.mining.connectors.base import BinaryTransportPort
from clingrounder.mining.connectors.http import UrllibBinaryTransport
from clingrounder.mining.connectors.local import LocalArchiveConnector
from clingrounder.mining.connectors.sources import (
    ClinicalTrialsConnector,
    DailyMedConnector,
    PmcOaConnector,
    StaticHttpConnector,
)
from clingrounder.mining.ports import SourceConnectorPort
from clingrounder.mining.registry import SourceDefinition

__all__ = ["connector_from_definition"]


def connector_from_definition(
    source: SourceDefinition,
    *,
    transport: BinaryTransportPort | None = None,
) -> SourceConnectorPort:
    """Instantiate a registered connector without importing task-specific code."""

    if source.connector == "local_archive":
        if transport is not None:
            raise ValueError("local_archive connectors do not accept an HTTP transport")
        return LocalArchiveConnector(source)
    resolved_transport = transport or UrllibBinaryTransport()
    if source.connector == "static_http":
        return StaticHttpConnector(source, resolved_transport)
    if source.connector == "pmc_oa":
        return PmcOaConnector(source, resolved_transport)
    if source.connector == "dailymed":
        return DailyMedConnector(source, resolved_transport)
    if source.connector == "clinicaltrials":
        return ClinicalTrialsConnector(source, resolved_transport)
    raise ValueError(
        f"Source {source.id!r} declares unsupported connector {source.connector!r}"
    )
