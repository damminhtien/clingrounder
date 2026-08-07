"""Deterministic discovery and readiness reporting for checked-in pipeline profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from clingrounder.pipeline.config_loader import ResolvedPipelineConfig
from clingrounder.pipeline.profile import PipelineProfileMetadata

__all__ = [
    "PipelineProfileCatalogEntry",
    "discover_pipeline_profiles",
    "inspect_pipeline_profiles",
    "validate_pipeline_profile_catalog",
]


@dataclass(frozen=True, slots=True)
class PipelineProfileCatalogEntry:
    """Inspection result for one profile, including failures without hiding them."""

    path: Path
    profile: PipelineProfileMetadata | None
    resources_ready: bool
    missing_resources: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self, *, root: Path | None = None) -> dict[str, object]:
        """Return a stable, machine-readable catalog row."""

        path = self.path
        if root is not None:
            try:
                path_value = str(path.relative_to(root))
            except ValueError:
                path_value = str(path)
        else:
            path_value = str(path)
        return {
            "path": path_value,
            "profile": None if self.profile is None else self.profile.to_dict(),
            "resources_ready": self.resources_ready,
            "missing_resources": list(self.missing_resources),
            "error": self.error,
        }


def discover_pipeline_profiles(root: str | Path = "configs/pipeline") -> tuple[Path, ...]:
    """Discover YAML profiles in stable path order without depending on cwd state."""

    profile_root = Path(root).expanduser().resolve()
    if not profile_root.is_dir():
        raise ValueError(f"Pipeline profile directory does not exist: {profile_root}")
    return tuple(sorted(profile_root.rglob("*.yaml"), key=lambda path: str(path)))


def inspect_pipeline_profiles(
    root: str | Path = "configs/pipeline",
) -> tuple[PipelineProfileCatalogEntry, ...]:
    """Load every profile and report resource readiness without constructing components."""

    entries: list[PipelineProfileCatalogEntry] = []
    for path in discover_pipeline_profiles(root):
        try:
            resolved = ResolvedPipelineConfig.load(path, require_profile=True)
            missing = tuple(
                str(resource["field"])
                for resource in resolved.inspection_report()["resources"]
                if not resource["exists"]
            )
            entries.append(
                PipelineProfileCatalogEntry(
                    path=path,
                    profile=resolved.profile,
                    resources_ready=not missing,
                    missing_resources=missing,
                )
            )
        except (OSError, TypeError, ValueError) as error:
            entries.append(
                PipelineProfileCatalogEntry(
                    path=path,
                    profile=None,
                    resources_ready=False,
                    error=f"{type(error).__name__}: {error}",
                )
            )
    return tuple(entries)


def validate_pipeline_profile_catalog(
    entries: tuple[PipelineProfileCatalogEntry, ...],
) -> tuple[str, ...]:
    """Return deterministic catalog errors for CI and profile-listing commands."""

    errors: list[str] = []
    ids: dict[str, Path] = {}
    for entry in entries:
        if entry.error is not None:
            errors.append(f"{entry.path}: {entry.error}")
            continue
        if entry.profile is None:
            errors.append(f"{entry.path}: missing profile metadata")
            continue
        previous = ids.get(entry.profile.profile_id)
        if previous is not None:
            errors.append(
                f"duplicate profile id {entry.profile.profile_id!r}: {previous} and {entry.path}"
            )
        else:
            ids[entry.profile.profile_id] = entry.path
        if entry.profile.portability.value == "portable" and not entry.resources_ready:
            errors.append(
                f"{entry.path}: portable profile has missing resources: "
                f"{', '.join(entry.missing_resources)}"
            )
    return tuple(sorted(errors))
