"""HTTP transport for source connectors using only the Python standard library."""

from __future__ import annotations

from typing import BinaryIO, cast
from urllib.request import Request, urlopen

__all__ = ["UrllibBinaryTransport"]


class UrllibBinaryTransport:
    """Stream public HTTP artifacts with an explicit, auditable user agent."""

    def __init__(
        self,
        *,
        user_agent: str = "clingrounder-data-miner/0.2",
        timeout_seconds: float = 60.0,
    ) -> None:
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds

    def open(self, uri: str) -> BinaryIO:
        request = Request(uri, headers={"User-Agent": self.user_agent})
        # SCALING: urlopen returns a streaming response; large archives are not buffered in RAM.
        return cast(BinaryIO, urlopen(request, timeout=self.timeout_seconds))  # noqa: S310
