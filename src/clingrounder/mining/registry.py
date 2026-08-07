"""Strict source-registry schema for licensed, versioned mining connectors."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from clingrounder.mining.records import AccessClass, RedistributionPolicy

__all__ = [
    "LicenseMode",
    "RetentionPolicy",
    "SourceDefinition",
    "SourceRegistry",
    "VersionPolicy",
    "load_source_registry",
]


class VersionPolicy(str, Enum):
    """How a connector resolves a source release."""

    PINNED = "pinned"
    LATEST_WITH_SNAPSHOT = "latest_with_snapshot"
    MANUAL_IMPORT = "manual_import"


class RetentionPolicy(str, Enum):
    """Whether raw bytes may be retained outside the current controlled environment."""

    IMMUTABLE = "immutable"
    REPLACEABLE = "replaceable"
    LOCAL_ONLY = "local_only"


class LicenseMode(str, Enum):
    """Whether license terms are source-wide or must be resolved for each artifact."""

    FIXED = "fixed"
    PER_ARTIFACT = "per_artifact"


class SourceDefinition(BaseModel):
    """Configuration and policy for one supported source connector."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    version: str = Field(min_length=1)
    version_policy: VersionPolicy
    access_class: AccessClass
    license_id: str = Field(min_length=1)
    license_url: str = Field(min_length=1)
    license_mode: LicenseMode = LicenseMode.FIXED
    redistribution: RedistributionPolicy
    hosted_processing_allowed: bool
    retention: RetentionPolicy
    connector: str = Field(min_length=1)
    parser: str = Field(min_length=1)
    parser_options: dict[str, str] = Field(default_factory=dict)
    urls: tuple[str, ...] = ()
    allowed_uses: tuple[str, ...]
    credentials: tuple[str, ...] = ()
    rate_limit_per_second: float | None = Field(default=None, gt=0.0)
    notes: str = ""

    @model_validator(mode="after")
    def validate_policy(self) -> "SourceDefinition":
        restricted = {
            AccessClass.CREDENTIALLED,
            AccessClass.DUA,
            AccessClass.LOCAL_PRIVATE,
            AccessClass.QUARANTINE,
        }
        if self.access_class in restricted and self.hosted_processing_allowed:
            raise ValueError("restricted sources cannot allow hosted processing")
        private_access = {
            AccessClass.AUTHORIZED_PRIVATE,
            AccessClass.DUA,
            AccessClass.LOCAL_PRIVATE,
        }
        if self.access_class in private_access:
            if self.retention is not RetentionPolicy.LOCAL_ONLY:
                raise ValueError("private sources require local_only canonical retention")
            if self.redistribution is not RedistributionPolicy.PROHIBITED:
                raise ValueError("private sources must prohibit redistribution")
        if (
            self.access_class is AccessClass.AUTHORIZED_PRIVATE
            and not self.hosted_processing_allowed
        ):
            raise ValueError(
                "authorized-private sources must explicitly allow hosted processing"
            )
        if self.access_class is AccessClass.QUARANTINE:
            if self.redistribution is not RedistributionPolicy.UNKNOWN:
                raise ValueError("quarantine sources must keep redistribution unknown")
        if not self.allowed_uses or any(not value.strip() for value in self.allowed_uses):
            raise ValueError("allowed_uses must contain non-empty values")
        if any(not key.strip() or not value.strip() for key, value in self.parser_options.items()):
            raise ValueError("parser_options must contain non-empty keys and values")
        if self.version_policy is VersionPolicy.PINNED and self.version.lower() == "latest":
            raise ValueError("pinned sources require an explicit version")
        return self


class SourceRegistry(BaseModel):
    """Versioned collection of unique source definitions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["medical-source-registry.v2"]
    resources: tuple[SourceDefinition, ...]

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "SourceRegistry":
        identifiers = [resource.id for resource in self.resources]
        duplicates = sorted({value for value in identifiers if identifiers.count(value) > 1})
        if duplicates:
            raise ValueError(f"duplicate source IDs: {', '.join(duplicates)}")
        return self

    def by_id(self, source_id: str) -> SourceDefinition:
        for resource in self.resources:
            if resource.id == source_id:
                return resource
        raise KeyError(f"Unknown mining source {source_id!r}")


def load_source_registry(path: str | Path) -> SourceRegistry:
    """Load and strictly validate a YAML source registry."""

    payload: Any
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return SourceRegistry.model_validate(payload)
