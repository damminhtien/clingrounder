"""Resumable orchestration for declarative mining plans."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from medical_kg_nlp.mining.connectors import connector_from_definition
from medical_kg_nlp.mining.io import (
    load_annotations,
    load_documents,
    load_relations,
    load_source_artifacts,
    write_json,
    write_jsonl,
)
from medical_kg_nlp.mining.parsers import parser_from_definition
from medical_kg_nlp.mining.policy import SourcePolicyGate
from medical_kg_nlp.mining.ports import ArtifactStorePort
from medical_kg_nlp.mining.records import (
    MinedDocument,
    SourceArtifact,
    SourceRequest,
)
from medical_kg_nlp.mining.registry import SourceDefinition, load_source_registry
from medical_kg_nlp.mining.snapshot import SnapshotBuilder, SnapshotSplitConfig
from medical_kg_nlp.mining.storage import FsspecArtifactStore, LocalArtifactStore

__all__ = [
    "MiningPlan",
    "MiningPlanResult",
    "SourceJob",
    "artifact_store_from_uri",
    "build_documents",
    "load_mining_plan",
    "run_mining_plan",
    "sync_source",
]


class ArtifactStoreConfig(BaseModel):
    """Content-addressed local path or fsspec URI."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    uri: str = Field(min_length=1)
    storage_options: dict[str, Any] = Field(default_factory=dict)


class SourceJob(BaseModel):
    """One versioned, cacheable source request in a mining plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    parse_documents: bool = True
    enabled: bool = True


class SnapshotPlan(BaseModel):
    """Optional final freeze stage for a mining run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)
    annotations: str | None = None
    relations: str | None = None
    development_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    challenge_sources: tuple[str, ...] = ()
    challenge_templates: tuple[str, ...] = ()
    hash_salt: str = "medical-kg-phase2-v1"
    max_synthetic_train_fraction: float = Field(default=0.4, ge=0.0, le=1.0)
    write_parquet: bool = True


class MiningPlan(BaseModel):
    """Strict schema for ``medical-kg data run --plan``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["medical-mining-plan.v1"]
    registry: str = "data/sources/mining_registry.yaml"
    work_dir: str = "outputs/mining/phase2"
    artifact_store: ArtifactStoreConfig
    sources: tuple[SourceJob, ...]
    snapshot: SnapshotPlan | None = None


class MiningPlanResult(BaseModel):
    """Machine-readable outputs and cache statistics from one plan run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    work_dir: str
    artifact_manifest: str
    document_manifest: str
    artifact_count: int
    document_count: int
    cache_hits: int
    cache_misses: int
    snapshot_id: str | None = None


def load_mining_plan(path: str | Path) -> MiningPlan:
    """Load a plan and resolve filesystem paths relative to its directory."""

    plan_path = Path(path).resolve()
    raw = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    plan = MiningPlan.model_validate(raw)
    base = plan_path.parent
    artifact_store_uri = _resolve_uri(plan.artifact_store.uri, base)
    snapshot = plan.snapshot
    resolved_snapshot = None
    if snapshot is not None:
        resolved_snapshot = snapshot.model_copy(
            update={
                "output_dir": str(_resolve_path(snapshot.output_dir, base)),
                "annotations": (
                    None
                    if snapshot.annotations is None
                    else str(_resolve_path(snapshot.annotations, base))
                ),
                "relations": (
                    None
                    if snapshot.relations is None
                    else str(_resolve_path(snapshot.relations, base))
                ),
            }
        )
    resolved_jobs = tuple(
        job.model_copy(update={"parameters": _resolve_job_parameters(job.parameters, base)})
        for job in plan.sources
    )
    return plan.model_copy(
        update={
            "registry": str(_resolve_path(plan.registry, base)),
            "work_dir": str(_resolve_path(plan.work_dir, base)),
            "artifact_store": plan.artifact_store.model_copy(
                update={"uri": artifact_store_uri}
            ),
            "sources": resolved_jobs,
            "snapshot": resolved_snapshot,
        }
    )


def run_mining_plan(path: str | Path) -> MiningPlanResult:
    """Execute acquisition, parsing, and optional freezing with stage-level resume."""

    plan = load_mining_plan(path)
    registry = load_source_registry(plan.registry)
    policy_gate = SourcePolicyGate(registry)
    store = artifact_store_from_uri(
        plan.artifact_store.uri,
        storage_options=plan.artifact_store.storage_options,
    )
    work_dir = Path(plan.work_dir)
    stage_root = work_dir / "stages"
    stage_root.mkdir(parents=True, exist_ok=True)
    all_artifacts: list[SourceArtifact] = []
    all_documents: list[MinedDocument] = []
    cache_hits = 0
    cache_misses = 0

    for job in plan.sources:
        if not job.enabled:
            continue
        source = registry.by_id(job.source_id)
        fingerprint = _job_fingerprint(source, job)
        stage_dir = stage_root / f"{job.source_id}-{fingerprint[:16]}"
        artifact_path = stage_dir / "artifacts.jsonl"
        document_path = stage_dir / "documents.jsonl"
        state_path = stage_dir / "state.json"
        state = _load_stage_state(state_path)
        if artifact_path.is_file() and state.get("acquisition") == "complete":
            artifacts = load_source_artifacts(artifact_path)
            cache_hits += 1
        else:
            stage_dir.mkdir(parents=True, exist_ok=True)
            artifacts = sync_source(
                source=source,
                request=SourceRequest(
                    source_id=job.source_id,
                    source_version=job.source_version,
                    parameters=job.parameters,
                ),
                store=store,
                policy_gate=policy_gate,
                checkpoint_path=artifact_path,
            )
            write_json(
                state_path,
                {**state, "acquisition": "complete"},
            )
            state = {**state, "acquisition": "complete"}
            cache_misses += 1
        all_artifacts.extend(artifacts)
        if not job.parse_documents:
            continue
        if document_path.is_file() and state.get("parsing") == "complete":
            documents = load_documents(document_path)
            cache_hits += 1
        else:
            documents = build_documents(
                source=source,
                artifacts=artifacts,
                store=store,
            )
            write_jsonl(
                document_path,
                (document.to_dict() for document in documents),
            )
            write_json(
                state_path,
                {**state, "parsing": "complete"},
            )
            cache_misses += 1
        all_documents.extend(documents)

    unique_artifacts = _unique_artifacts(all_artifacts)
    unique_documents = _unique_documents(all_documents)
    work_dir.mkdir(parents=True, exist_ok=True)
    artifact_manifest = work_dir / "artifacts.jsonl"
    document_manifest = work_dir / "documents.jsonl"
    write_jsonl(
        artifact_manifest,
        (artifact.to_dict() for artifact in unique_artifacts),
    )
    write_jsonl(
        document_manifest,
        (document.to_dict() for document in unique_documents),
    )

    snapshot_id = None
    if plan.snapshot is not None:
        snapshot_plan = plan.snapshot
        annotations = (
            ()
            if snapshot_plan.annotations is None
            else load_annotations(snapshot_plan.annotations)
        )
        relations = (
            () if snapshot_plan.relations is None else load_relations(snapshot_plan.relations)
        )
        builder = SnapshotBuilder(
            split_config=SnapshotSplitConfig(
                development_fraction=snapshot_plan.development_fraction,
                challenge_sources=frozenset(snapshot_plan.challenge_sources),
                challenge_templates=frozenset(snapshot_plan.challenge_templates),
                hash_salt=snapshot_plan.hash_salt,
                max_synthetic_train_fraction=(
                    snapshot_plan.max_synthetic_train_fraction
                ),
            )
        )
        snapshot = builder.freeze(
            version=snapshot_plan.version,
            created_at=snapshot_plan.created_at,
            output_dir=snapshot_plan.output_dir,
            documents=unique_documents,
            annotations=annotations,
            relations=relations,
            source_fingerprints=tuple(
                artifact.object.sha256 for artifact in unique_artifacts
            ),
            write_parquet=snapshot_plan.write_parquet,
        )
        snapshot_id = snapshot.snapshot_id

    result = MiningPlanResult(
        work_dir=str(work_dir),
        artifact_manifest=str(artifact_manifest),
        document_manifest=str(document_manifest),
        artifact_count=len(unique_artifacts),
        document_count=len(unique_documents),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        snapshot_id=snapshot_id,
    )
    write_json(work_dir / "run_result.json", result.model_dump(mode="json"))
    return result


def sync_source(
    *,
    source: SourceDefinition,
    request: SourceRequest,
    store: ArtifactStorePort,
    policy_gate: SourcePolicyGate,
    checkpoint_path: str | Path | None = None,
) -> tuple[SourceArtifact, ...]:
    """Fetch one source request and checkpoint every completed artifact."""

    connector = connector_from_definition(source)
    completed = (
        list(load_source_artifacts(checkpoint_path))
        if checkpoint_path is not None and Path(checkpoint_path).is_file()
        else []
    )
    completed_by_uri = {artifact.source_uri: artifact for artifact in completed}
    discovered_count = 0
    for discovered in connector.discover(request):
        discovered_count += 1
        artifact = completed_by_uri.get(discovered.uri)
        if (
            artifact is not None
            and discovered.expected_sha256 is not None
            and artifact.object.sha256 != discovered.expected_sha256
        ):
            artifact = None
        if artifact is None:
            artifact = connector.fetch(discovered, store=store)
        decision = policy_gate.validate_artifact(artifact)
        if not decision.allowed:
            raise PermissionError(
                f"Artifact policy rejected {artifact.artifact_id}: {', '.join(decision.reasons)}"
            )
        completed_by_uri[discovered.uri] = artifact
        completed = list(completed_by_uri.values())
        if checkpoint_path is not None:
            # SCALING: each completed artifact is durable, so interruption resumes at CAS speed.
            write_jsonl(
                checkpoint_path,
                (
                    value.to_dict()
                    for value in sorted(completed, key=lambda item: item.artifact_id)
                ),
            )
    if discovered_count == 0:
        raise ValueError(f"Source request {request.source_id!r} discovered no artifacts")
    return tuple(sorted(completed, key=lambda item: item.artifact_id))


def build_documents(
    *,
    source: SourceDefinition,
    artifacts: Sequence[SourceArtifact],
    store: ArtifactStorePort,
) -> tuple[MinedDocument, ...]:
    """Parse artifacts through the source-declared parser and reject duplicate IDs."""

    parser = parser_from_definition(source)
    documents = tuple(
        document
        for artifact in sorted(artifacts, key=lambda item: item.artifact_id)
        for document in parser.parse(artifact, store=store)
    )
    return _unique_documents(documents)


def artifact_store_from_uri(
    uri: str, *, storage_options: Mapping[str, Any] | None = None
) -> ArtifactStorePort:
    """Build a local or optional fsspec store without leaking backend choice to commands."""

    if "://" in uri and not uri.startswith("file://"):
        return FsspecArtifactStore(
            uri,
            storage_options=storage_options,
        )
    local_path = uri.removeprefix("file://")
    return LocalArtifactStore(local_path)


def _job_fingerprint(source: SourceDefinition, job: SourceJob) -> str:
    connector = connector_from_definition(source)
    parser_revision = None
    if job.parse_documents:
        parser = parser_from_definition(source)
        parser_revision = getattr(parser, "parser_revision", None)
    payload = {
        "source": source.model_dump(mode="json"),
        "job": job.model_dump(mode="json"),
        "connector_revision": getattr(connector, "connector_revision", None),
        "parser_revision": parser_revision,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _unique_artifacts(values: Sequence[SourceArtifact]) -> tuple[SourceArtifact, ...]:
    by_id: dict[str, SourceArtifact] = {}
    for value in values:
        previous = by_id.setdefault(value.artifact_id, value)
        if previous != value:
            raise ValueError(f"Conflicting artifact ID {value.artifact_id!r}")
    return tuple(sorted(by_id.values(), key=lambda item: item.artifact_id))


def _unique_documents(values: Sequence[MinedDocument]) -> tuple[MinedDocument, ...]:
    by_id: dict[str, MinedDocument] = {}
    for value in values:
        previous = by_id.setdefault(value.document_id, value)
        if previous != value:
            raise ValueError(f"Conflicting document ID {value.document_id!r}")
    return tuple(sorted(by_id.values(), key=lambda item: item.document_id))


def _resolve_job_parameters(parameters: Mapping[str, Any], base: Path) -> dict[str, Any]:
    result = dict(parameters)
    paths = result.get("paths")
    if isinstance(paths, Sequence) and not isinstance(paths, (str, bytes)):
        result["paths"] = [str(_resolve_path(str(path), base)) for path in paths]
    return result


def _resolve_path(value: str, base: Path) -> Path:
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise ValueError(f"Unresolved environment variable in path {value!r}")
    path = Path(expanded).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _resolve_uri(value: str, base: Path) -> str:
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise ValueError(f"Unresolved environment variable in URI {value!r}")
    if "://" in expanded:
        return expanded
    return str(_resolve_path(expanded, base))


def _load_stage_state(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in raw.items()
    ):
        raise ValueError(f"Invalid mining stage state: {path}")
    return raw
