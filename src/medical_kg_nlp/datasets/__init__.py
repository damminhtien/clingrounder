"""Dataset adapters that convert source records into core document schemas."""

from __future__ import annotations

from medical_kg_nlp.datasets.base import DatasetAdapter
from medical_kg_nlp.datasets.synthetic_adapter import SyntheticDatasetAdapter

__all__ = ["DatasetAdapter", "SyntheticDatasetAdapter"]
