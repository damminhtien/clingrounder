from __future__ import annotations

import pytest

from medical_kg_nlp.adapters import HuggingFaceCrossEncoderAdapter, HuggingFaceModelConfig
from medical_kg_nlp.pipeline import (
    PipelineFactory,
    PipelineConfig,
    PipelineOptions,
    PipelineRuntime,
)


class _Resource:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    def close(self) -> None:
        self.events.append(self.name)


class _Runner(_Resource):
    pass


def test_pipeline_runtime_closes_in_reverse_order_and_is_idempotent() -> None:
    events: list[str] = []
    first = _Resource("first", events)
    second = _Resource("second", events)
    runner = _Runner("runner", events)
    runtime = PipelineRuntime(runner, (first, second))

    with runtime as active:
        assert active.runner is runner
    runtime.close()

    assert events == ["second", "first", "runner"]
    with pytest.raises(RuntimeError, match="closed"):
        _ = runtime.runner


def test_pipeline_runtime_closes_resources_after_one_close_failure() -> None:
    events: list[str] = []

    class _FailingResource(_Resource):
        def close(self) -> None:
            events.append(self.name)
            raise OSError("close failed")

    runtime = PipelineRuntime(
        _Runner("runner", events),
        (_Resource("first", events), _FailingResource("failing", events)),
    )

    with pytest.raises(RuntimeError, match="failed to close"):
        runtime.close()
    runtime.close()
    assert events == ["failing", "first", "runner"]


def test_pipeline_factory_exposes_managed_runtime() -> None:
    runtime = PipelineFactory.runtime_from_config(
        PipelineConfig(
            options=PipelineOptions(
                enable_context=False,
                enable_linking=False,
                enable_candidate_reranking=False,
                enable_entity_kg_validation=False,
                enable_relations=False,
                enable_relation_kg_validation=False,
            )
        )
    )
    try:
        assert runtime.runner.process_text("runtime", "Bệnh nhân ho.").document_id == "runtime"
    finally:
        runtime.close()


def test_model_adapter_cleanup_hook_releases_loaded_model() -> None:
    adapter = HuggingFaceCrossEncoderAdapter(
        HuggingFaceModelConfig(model_id="local/model", revision="revision")
    )

    class _Model:
        def __init__(self) -> None:
            self.cpu_calls = 0

        def cpu(self) -> None:
            self.cpu_calls += 1

    model = _Model()
    adapter._loaded = (object(), object(), model)
    adapter.close()
    adapter.close()

    assert model.cpu_calls == 1
    assert adapter._loaded is None
