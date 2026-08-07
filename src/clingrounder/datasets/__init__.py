"""Dataset adapters that convert source records into core document schemas."""

from __future__ import annotations

from clingrounder.datasets.base import DatasetAdapter
from clingrounder.datasets.synthetic_adapter import SyntheticDatasetAdapter

__all__ = ["DatasetAdapter", "SyntheticDatasetAdapter"]
