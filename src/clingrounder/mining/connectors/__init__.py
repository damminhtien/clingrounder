"""Licensed, checkpointable source acquisition adapters."""

from clingrounder.mining.connectors.base import (
    BinaryTransportPort,
    RegisteredConnectorAdapter,
)
from clingrounder.mining.connectors.factory import connector_from_definition
from clingrounder.mining.connectors.http import UrllibBinaryTransport
from clingrounder.mining.connectors.local import LocalArchiveConnector, LocalFileTransport
from clingrounder.mining.connectors.sources import (
    ClinicalTrialsConnector,
    DailyMedConnector,
    PmcOaConnector,
    StaticHttpConnector,
)

__all__ = [
    "BinaryTransportPort",
    "ClinicalTrialsConnector",
    "DailyMedConnector",
    "LocalArchiveConnector",
    "LocalFileTransport",
    "PmcOaConnector",
    "RegisteredConnectorAdapter",
    "StaticHttpConnector",
    "UrllibBinaryTransport",
    "connector_from_definition",
]
