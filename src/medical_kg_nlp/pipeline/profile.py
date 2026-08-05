"""Self-describing metadata for reusable pipeline profiles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "PIPELINE_PROFILE_SCHEMA_VERSION",
    "PipelineProfileMetadata",
    "ProfilePortability",
    "ProfileMaturity",
    "ProfileSupportStatus",
    "migrate_pipeline_profile",
]

PIPELINE_PROFILE_SCHEMA_VERSION = "medical-kg.pipeline-profile.v2"


class ProfileMaturity(str, Enum):
    """Lifecycle state advertised by a checked-in pipeline profile."""

    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    BENCHMARK = "benchmark"


class ProfilePortability(str, Enum):
    """Where a profile is expected to work and what setup it may require."""

    PORTABLE = "portable"
    TEMPLATE = "template"
    LOCAL = "local"
    EXPERIMENTAL = "experimental"
    ARCHIVED = "archived"


class ProfileSupportStatus(str, Enum):
    """Support promise made by the repository for a profile."""

    SUPPORTED = "supported"
    SETUP_REQUIRED = "setup_required"
    EXPERIMENTAL = "experimental"
    ARCHIVED = "archived"


@dataclass(frozen=True, slots=True)
class PipelineProfileMetadata:
    """Human-facing identity and dependency contract for one pipeline profile."""

    profile_id: str
    title: str
    description: str
    maturity: ProfileMaturity
    required_extras: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    portability: ProfilePortability = ProfilePortability.PORTABLE
    support_status: ProfileSupportStatus = ProfileSupportStatus.SUPPORTED
    owner: str = "medical-kg-nlp"

    def __post_init__(self) -> None:
        for field_name, value in (
            ("id", self.profile_id),
            ("title", self.title),
            ("description", self.description),
            ("owner", self.owner),
        ):
            if not value.strip():
                raise ValueError(f"profile.{field_name} must be non-empty")
        if any(not value.strip() for value in (*self.required_extras, *self.tags)):
            raise ValueError("profile extras and tags must be non-empty strings")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "PipelineProfileMetadata":
        """Parse metadata and reject misspelled or undocumented profile fields."""

        allowed = {
            "id",
            "title",
            "description",
            "maturity",
            "required_extras",
            "tags",
            "portability",
            "support_status",
            "owner",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown pipeline profile keys: {', '.join(unknown)}")
        return cls(
            profile_id=_required_string(payload, "id"),
            title=_required_string(payload, "title"),
            description=_required_string(payload, "description"),
            maturity=ProfileMaturity(_required_string(payload, "maturity")),
            required_extras=_string_tuple(payload, "required_extras"),
            tags=_string_tuple(payload, "tags"),
            portability=ProfilePortability(
                str(payload.get("portability", ProfilePortability.PORTABLE.value))
            ),
            support_status=ProfileSupportStatus(
                str(payload.get("support_status", ProfileSupportStatus.SUPPORTED.value))
            ),
            owner=(
                _required_string(payload, "owner")
                if "owner" in payload
                else "medical-kg-nlp"
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON/YAML-ready metadata."""

        return {
            "id": self.profile_id,
            "title": self.title,
            "description": self.description,
            "maturity": self.maturity.value,
            "required_extras": list(self.required_extras),
            "tags": list(self.tags),
            "portability": self.portability.value,
            "support_status": self.support_status.value,
            "owner": self.owner,
        }


def migrate_pipeline_profile(payload: Mapping[str, object]) -> dict[str, object]:
    """Return a deterministic current-schema copy or reject unsupported versions.

    There are no historical migrations yet. Keeping this explicit prevents a loader from
    silently accepting a profile with incompatible semantics.
    """

    version = payload.get("schema_version")
    if version != PIPELINE_PROFILE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported pipeline profile schema_version: "
            f"{version!r}; expected {PIPELINE_PROFILE_SCHEMA_VERSION!r}"
        )
    return dict(payload)


def _required_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"profile.{key} must be a non-empty string")
    return value


def _string_tuple(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key, ())
    if not isinstance(value, list | tuple) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"profile.{key} must be an array of non-empty strings")
    return tuple(value)
