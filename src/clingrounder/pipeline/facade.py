"""Small lifecycle-aware API for ordinary users of the clinical NLP toolkit."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
from pathlib import Path

from clingrounder.pipeline.components import PipelineComponents
from clingrounder.pipeline.config_loader import ResolvedPipelineConfig
from clingrounder.pipeline.factory import PipelineFactory, PipelineConfig
from clingrounder.pipeline.parallel_batch import ParallelBatchOptions, PipelineBatchExecutor
from clingrounder.pipeline.profile_catalog import inspect_pipeline_profiles
from clingrounder.pipeline.runner import PipelineRunResult, PipelineRunner
from clingrounder.pipeline.runtime import PipelineRuntime
from clingrounder.schema.document import ClinicalDocument
from clingrounder.schema.output import ClinicalPrediction
from clingrounder.artifacts.registry import BuiltinArtifact, get_builtin_artifact

__all__ = [
    "Pipeline",
    "PipelineClosedError",
    "PipelineConfig",
    "PipelineConfigurationError",
    "UnknownProfileError",
    "load_pipeline",
]

RunnerFactory = Callable[[], PipelineRunner]


class UnknownProfileError(LookupError):
    """Raised when a named profile cannot be found in the repository catalog."""


class PipelineConfigurationError(ValueError):
    """Raised when a profile exists but cannot be composed on this machine."""


class PipelineClosedError(RuntimeError):
    """Raised when an operation is attempted after a pipeline has been closed."""


class Pipeline:
    """The ordinary user entry point for one composed clinical NLP runtime."""

    def __init__(
        self,
        runtime: PipelineRuntime,
        runner_factory: RunnerFactory | None = None,
    ) -> None:
        self._runtime = runtime
        self._runner_factory = runner_factory
        self._closed = False

    @classmethod
    def from_profile(cls, profile: str) -> "Pipeline":
        """Load a repository-known profile by ID without using the caller's cwd."""

        if not profile.strip():
            raise UnknownProfileError("Pipeline profile name must be non-empty")
        root = _repository_profile_root()
        entries = inspect_pipeline_profiles(root)
        matches = [
            entry.path
            for entry in entries
            if entry.profile is not None and entry.profile.profile_id == profile
        ]
        if not matches:
            raise UnknownProfileError(
                f"Unknown pipeline profile {profile!r}; searched {root}"
            )
        if len(matches) > 1:
            raise PipelineConfigurationError(
                f"Ambiguous pipeline profile {profile!r}: {', '.join(map(str, matches))}"
            )
        return cls.from_config(matches[0])

    @classmethod
    def from_config(cls, path: str | Path) -> "Pipeline":
        """Load and compose one self-describing YAML profile."""

        config_path = Path(path).expanduser().resolve()
        try:
            resolved = ResolvedPipelineConfig.load(config_path, require_profile=True)
            factory_config = resolved.factory_config
            runtime = PipelineFactory.runtime_from_config(factory_config)
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            raise PipelineConfigurationError(
                f"Unable to compose pipeline profile {config_path}: {error}"
            ) from error
        return cls(runtime, partial(PipelineFactory.from_config, factory_config))

    @classmethod
    def from_components(cls, components: PipelineComponents) -> "Pipeline":
        """Create a pipeline from advanced components without loading config or files."""

        runner = PipelineRunner(components)
        return cls(PipelineRuntime(runner, runner.resources))

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        *,
        revision: str | None = None,
        offline: bool = False,
    ) -> "Pipeline":
        """Load a pinned package-bundled resource pack without network fallback.

        External model and terminology releases can be composed through ``from_config``.  The
        built-in pack is intentionally small so a clean installation has one deterministic,
        offline smoke path.
        """

        del offline  # Bundled artifacts are always local; no network fallback is permitted.
        artifact = get_builtin_artifact(name, revision)
        config = _builtin_artifact_config(artifact)
        runtime = PipelineFactory.runtime_from_config(config)
        return cls(runtime, partial(PipelineFactory.from_config, config))

    @classmethod
    def download(
        cls,
        name: str,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        offline: bool = False,
    ) -> Path:
        """Materialize a pinned built-in pack into a caller-owned cache.

        This method does not contact a remote registry.  A future remote registry must be
        implemented as a separately reviewed artifact provider with checksum verification.
        """

        del offline
        artifact = get_builtin_artifact(name, revision)
        destination = Path(cache_dir or Path.home() / ".cache" / "clingrounder" / "artifacts")
        return artifact.install(destination)

    def predict(
        self,
        text: str,
        *,
        document_id: str,
        metadata: Mapping[str, str] | None = None,
    ) -> ClinicalPrediction:
        """Predict entities, assertions, links, and relations for one text."""

        self._ensure_open()
        return self._runtime.runner.process_text(
            document_id,
            text,
            dict(metadata) if metadata is not None else None,
        )

    def __call__(
        self,
        text: str,
        *,
        document_id: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> ClinicalPrediction:
        """Predict with a deterministic content-derived ID for short examples."""

        resolved_id = document_id or f"text-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}"
        return self.predict(text, document_id=resolved_id, metadata=metadata)

    def predict_document(self, document: ClinicalDocument) -> ClinicalPrediction:
        """Predict from an already validated clinical document."""

        self._ensure_open()
        return self._runtime.runner.process_document(document)

    def predict_with_trace(
        self,
        text: str,
        *,
        document_id: str,
        metadata: Mapping[str, str] | None = None,
    ) -> PipelineRunResult:
        """Predict one text and retain the stage trace for diagnostics."""

        self._ensure_open()
        return self._runtime.runner.process_text_with_trace(
            document_id,
            text,
            dict(metadata) if metadata is not None else None,
        )

    def predict_many(
        self,
        documents: Sequence[ClinicalDocument],
        *,
        workers: int = 1,
    ) -> list[ClinicalPrediction]:
        """Predict documents in input order, optionally using independent workers."""

        self._ensure_open()
        if workers < 1:
            raise ValueError("workers must be at least 1")
        if workers == 1 or len(documents) < 2:
            return [self.predict_document(document) for document in documents]

        if self._runner_factory is not None:
            options = ParallelBatchOptions(
                backend="thread",
                max_workers=workers,
                chunksize=1,
                runtime_capabilities=self._runtime.runner.runtime_capabilities,
            )
            with PipelineBatchExecutor(self._runner_factory, options) as executor:
                results = executor.run(documents)
            return [result.prediction for result in results]

        capabilities = self._runtime.runner.runtime_capabilities
        if not capabilities.thread_safe:
            raise ValueError(
                "from_components pipelines require thread_safe components for workers > 1"
            )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Executor.map preserves input ordering while the runner remains the sole owner of
            # component state. The capability declaration is the concurrency contract.
            return list(executor.map(self.predict_document, documents))

    def close(self) -> None:
        """Close all owned resources; repeated calls are safe."""

        if self._closed:
            return
        self._closed = True
        self._runtime.close()

    def __enter__(self) -> "Pipeline":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise PipelineClosedError("Pipeline is closed")


def _repository_profile_root() -> Path:
    """Locate checked-in profiles from the installed source tree, never from cwd."""

    return Path(__file__).resolve().parents[3] / "configs" / "pipeline"


def _builtin_artifact_config(artifact: BuiltinArtifact) -> PipelineConfig:
    """Build a minimal config from package resources without writing into site-packages."""

    root = artifact.root
    cache_dir = Path.home() / ".cache" / "clingrounder" / "terminology"
    return PipelineConfig.from_mapping(
        {
            "terminology": {
                "recognition_path": str(root / "seed_concepts.jsonl"),
                "abbreviation_path": str(root / "abbreviations.jsonl"),
                "alias_overlay_path": str(root / "vietnamese_medical_alias.jsonl"),
                "cache_dir": str(cache_dir),
            },
            "pipeline": {
                "version": f"{artifact.artifact_id}-{artifact.revision}",
                "enable_context": True,
                "enable_linking": True,
                "enable_candidate_reranking": False,
                "enable_graph_evidence_reranking": False,
                "enable_entity_kg_validation": False,
                "enable_relations": False,
                "enable_relation_kg_validation": False,
                "max_candidates": 5,
                "candidate_sources": ["exact", "abbreviation"],
            },
        }
    )


def load_pipeline(
    name_or_path: str | Path,
    *,
    revision: str | None = None,
    offline: bool = False,
) -> Pipeline:
    """Load a repository profile or a pinned built-in artifact."""

    path = Path(name_or_path).expanduser()
    if path.exists():
        return Pipeline.from_config(path)
    return Pipeline.from_pretrained(str(name_or_path), revision=revision, offline=offline)
