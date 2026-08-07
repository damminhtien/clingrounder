"""Contracts for task-specific benchmark plugins."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

__all__ = ["BenchmarkHandler", "BenchmarkPlugin"]


@dataclass(frozen=True, slots=True)
class BenchmarkHandler:
    """Lazy import target for one benchmark CLI operation."""

    module: str
    function: str


class BenchmarkPlugin(Protocol):
    """Extension point implemented by built-in or third-party benchmark packages."""

    name: str
    summary: str

    def register_cli(
        self,
        parsers: argparse._SubParsersAction[argparse.ArgumentParser],
    ) -> None:
        """Attach plugin-specific arguments below ``clingrounder benchmark``."""

    def handlers(self) -> Mapping[str, BenchmarkHandler]:
        """Return lazy handler targets owned by this plugin."""
