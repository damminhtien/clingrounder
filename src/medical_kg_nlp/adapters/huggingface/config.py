"""Pinned configuration shared by local Hugging Face adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath

__all__ = ["HuggingFaceModelConfig"]


@dataclass(frozen=True)
class HuggingFaceModelConfig:
    """Pinned local model identity and bounded inference settings."""

    model_id: str
    revision: str
    device: str = "cpu"
    batch_size: int = 16
    max_length: int = 512
    subfolder: str | None = None

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must be a non-empty pinned model identifier")
        if not self.revision.strip():
            raise ValueError("revision must be a non-empty model revision")
        if not self.device.strip():
            raise ValueError("device must be non-empty")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.max_length < 8:
            raise ValueError("max_length must be at least 8")
        if self.subfolder is not None:
            subfolder = PurePosixPath(self.subfolder)
            if (
                not self.subfolder.strip()
                or subfolder.is_absolute()
                or ".." in subfolder.parts
            ):
                raise ValueError("subfolder must be a non-empty relative model path")

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        *,
        name: str,
    ) -> "HuggingFaceModelConfig":
        """Parse one model block and reject unpinned or malformed configurations."""

        model_id = payload.get("model_id")
        revision = payload.get("revision")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError(f"models.{name}.model_id is required")
        if not isinstance(revision, str) or not revision:
            raise ValueError(f"models.{name}.revision is required")
        return cls(
            model_id=model_id,
            revision=revision,
            device=_optional_string(payload, "device", "cpu", name),
            batch_size=_optional_int(payload, "batch_size", 16, name),
            max_length=_optional_int(payload, "max_length", 512, name),
            subfolder=_optional_nullable_string(payload, "subfolder", name),
        )

    @property
    def provenance(self) -> str:
        """Return a stable identity suitable for traces and experiment manifests."""

        suffix = "" if self.subfolder is None else f"#{self.subfolder}"
        return f"{self.model_id}@{self.revision}{suffix}"


def _optional_string(
    payload: Mapping[str, object],
    key: str,
    default: str,
    name: str,
) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"models.{name}.{key} must be a non-empty string")
    return value


def _optional_int(
    payload: Mapping[str, object],
    key: str,
    default: int,
    name: str,
) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"models.{name}.{key} must be an integer")
    return value


def _optional_nullable_string(
    payload: Mapping[str, object],
    key: str,
    name: str,
) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"models.{name}.{key} must be a non-empty string")
    return value
