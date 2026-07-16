"""Licensed, checkpointable source acquisition adapters."""

from medical_kg_nlp.mining.connectors.base import (
    BinaryTransportPort,
    RegisteredConnectorAdapter,
)
from medical_kg_nlp.mining.connectors.factory import connector_from_definition
from medical_kg_nlp.mining.connectors.http import UrllibBinaryTransport
from medical_kg_nlp.mining.connectors.local import LocalArchiveConnector, LocalFileTransport
from medical_kg_nlp.mining.connectors.sources import (
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
