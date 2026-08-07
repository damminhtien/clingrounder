"""Machine-readable status records for source-specific mining work.

The source registry answers whether data may be acquired.  This module answers a different
question: how far a registered source has actually progressed through parsing, curation, and
knowledge promotion.  Keeping the two contracts separate prevents a registered connector from
being mistaken for a completed dataset.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from clingrounder.mining.registry import SourceRegistry

__all__ = [
    "PromotionBoundary",
    "SourceProcessingIndex",
    "SourceProcessingRecord",
    "SourceProcessingState",
    "load_source_processing_index",
    "validate_source_processing_paths",
]


class SourceProcessingState(str, Enum):
    """Furthest reproducible stage completed for one source."""

    REGISTERED = "registered"
    ACQUIRED = "acquired"
    PROPOSED = "proposed"
    CURATED = "curated"
    PROMOTED = "promoted"
    QUARANTINED = "quarantined"


class PromotionBoundary(str, Enum):
    """Strongest permitted use of derived knowledge from a source."""

    NONE = "none"
    REVIEW_ONLY = "review_only"
    TRAINING_ONLY = "training_only"
    RUNTIME_OPT_IN = "runtime_opt_in"
    RUNTIME_DEFAULT = "runtime_default"


class SourceProcessingRecord(BaseModel):
    """Discoverability and promotion state for one registered source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    state: SourceProcessingState
    promotion_boundary: PromotionBoundary
    dossier: str = Field(min_length=1)
    verified_on: date
    summary: str = Field(min_length=1)
    run_configs: tuple[str, ...] = ()
    artifact_roots: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_state_boundary(self) -> "SourceProcessingRecord":
        if self.state is SourceProcessingState.PROMOTED:
            if self.promotion_boundary in {
                PromotionBoundary.NONE,
                PromotionBoundary.REVIEW_ONLY,
                PromotionBoundary.TRAINING_ONLY,
            }:
                raise ValueError("promoted sources require a runtime promotion boundary")
        if self.state is SourceProcessingState.QUARANTINED:
            if self.promotion_boundary is not PromotionBoundary.NONE:
                raise ValueError("quarantined sources cannot promote derived knowledge")
        return self


class SourceProcessingIndex(BaseModel):
    """Versioned processing records whose source IDs must be registry-backed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["medical-source-processing.v1"]
    sources: tuple[SourceProcessingRecord, ...]

    @model_validator(mode="after")
    def validate_unique_sources(self) -> "SourceProcessingIndex":
        source_ids = [record.source_id for record in self.sources]
        duplicates = sorted(
            {source_id for source_id in source_ids if source_ids.count(source_id) > 1}
        )
        if duplicates:
            raise ValueError(f"duplicate processing source IDs: {', '.join(duplicates)}")
        return self


def load_source_processing_index(path: str | Path) -> SourceProcessingIndex:
    """Load the strict source-processing index without inspecting large artifacts."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return SourceProcessingIndex.model_validate(yaml.safe_load(handle))


def validate_source_processing_paths(
    index: SourceProcessingIndex,
    registry: SourceRegistry,
    *,
    repository_root: str | Path,
) -> tuple[str, ...]:
    """Return stable discoverability errors for docs and reproducible run configs.

    Artifact roots are intentionally not checked here because the canonical bytes may live on an
    external volume or S3.  Dossiers and run configurations are repository-owned contracts and
    therefore must remain available in a clean checkout.
    """

    root = Path(repository_root)
    registered_ids = {source.id for source in registry.resources}
    errors: list[str] = []
    for record in index.sources:
        if record.source_id not in registered_ids:
            errors.append(f"unknown_source:{record.source_id}")
        if not (root / record.dossier).is_file():
            errors.append(f"missing_dossier:{record.source_id}:{record.dossier}")
        for config in record.run_configs:
            if not (root / config).is_file():
                errors.append(f"missing_run_config:{record.source_id}:{config}")
    return tuple(sorted(errors))
