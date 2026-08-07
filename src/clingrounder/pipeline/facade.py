"""Small lifecycle-aware API for ordinary users of the clinical NLP toolkit."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
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

__all__ = [
    "Pipeline",
    "PipelineClosedError",
    "PipelineConfig",
    "PipelineConfigurationError",
    "UnknownProfileError",
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
