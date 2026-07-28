"""Fast contracts for structured local generative-model adapters."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from medical_kg_nlp.adapters.generative import (
    InferenceBudgetManifest,
    LocalPeftAdapterConfig,
    ModelBudgetEntry,
    StructuredResponseError,
    TransformersCausalLMRuntime,
    parse_structured_response,
)
from medical_kg_nlp.adapters.generative import runtime as generative_runtime
from medical_kg_nlp.utils.hashing import sha256_directory

_REVISION = "1" * 40


def test_model_budget_counts_distinct_checkpoints_not_repeated_passes() -> None:
    manifest = InferenceBudgetManifest(
        entries=(
            ModelBudgetEntry(
                artifact_id="qwen3-8b",
                model_id="Qwen/Qwen3-8B",
                revision=_REVISION,
                parameter_count=8_200_000_000,
                kind="base",
                roles=("adjudication", "recall", "targeted"),
            ),
        )
    )

    assert manifest.total_parameters == 8_200_000_000
    assert manifest.to_dict()["remaining_parameters"] == 800_000_000


def test_model_budget_rejects_combined_qwen_and_xlmr_above_limit() -> None:
    with pytest.raises(ValueError, match="budget exceeded"):
        InferenceBudgetManifest(
            entries=(
                ModelBudgetEntry(
                    artifact_id="qwen3-8b",
                    model_id="Qwen/Qwen3-8B",
                    revision=_REVISION,
                    parameter_count=8_200_000_000,
                    kind="base",
                    roles=("recall",),
                ),
                ModelBudgetEntry(
                    artifact_id="xlmr",
                    model_id="FacebookAI/xlm-roberta-base",
                    revision="2" * 40,
                    parameter_count=278_000_000,
                    kind="auxiliary",
                    roles=("verifier",),
                ),
                ModelBudgetEntry(
                    artifact_id="other",
                    model_id="example/other",
                    revision="3" * 40,
                    parameter_count=600_000_000,
                    kind="auxiliary",
                    roles=("reranker",),
                ),
            )
        )


def test_peft_runtime_loads_verified_adapter_without_merging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter")
    adapter = LocalPeftAdapterConfig(
        path=adapter_dir,
        fingerprint=sha256_directory(adapter_dir),
        parameter_count=12,
    )
    calls: dict[str, Any] = {}

    class _BaseModel:
        def to(self, device: str) -> None:
            calls["base_to"] = device

        def eval(self) -> None:
            calls["base_eval"] = True

    class _PeftModel(_BaseModel):
        def merge_and_unload(self) -> None:
            raise AssertionError("Inference must not merge PEFT weights")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> object:
            calls["tokenizer"] = (args, kwargs)
            return object()

    class _AutoModel:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> _BaseModel:
            calls["base"] = (args, kwargs)
            return _BaseModel()

    class _PeftFactory:
        @staticmethod
        def from_pretrained(
            model: _BaseModel,
            path: str,
            **kwargs: Any,
        ) -> _PeftModel:
            calls["adapter"] = (model, path, kwargs)
            return _PeftModel()

    class _AdapterTensor:
        def numel(self) -> int:
            return 12

    fake_torch = SimpleNamespace(
        bfloat16="bf16",
        float16="fp16",
        float32="fp32",
    )
    fake_transformers = SimpleNamespace(
        AutoTokenizer=_AutoTokenizer,
        AutoModelForCausalLM=_AutoModel,
    )
    fake_peft = SimpleNamespace(
        PeftModel=_PeftFactory,
        get_peft_model_state_dict=lambda model: {"lora.weight": _AdapterTensor()},
    )
    monkeypatch.setattr(
        generative_runtime,
        "_import_generative_dependencies",
        lambda: (fake_torch, fake_transformers),
    )
    monkeypatch.setattr(
        generative_runtime,
        "_import_peft_dependency",
        lambda: fake_peft,
    )
    runtime = TransformersCausalLMRuntime(
        model_id="Qwen/Qwen3-8B",
        revision=_REVISION,
        device="cpu",
        adapter=adapter,
    )

    assert calls == {}
    _, _, model = runtime._load()

    assert isinstance(model, _PeftModel)
    assert calls["adapter"][1] == str(adapter_dir)
    assert calls["adapter"][2] == {
        "is_trainable": False,
        "local_files_only": True,
    }
    assert calls["base_to"] == "cpu"
    assert calls["base_eval"] is True


def test_peft_runtime_rejects_declared_parameter_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    adapter = LocalPeftAdapterConfig(
        path=adapter_dir,
        fingerprint=sha256_directory(adapter_dir),
        parameter_count=13,
    )

    class _Model:
        def eval(self) -> None:
            pass

    class _Tensor:
        def numel(self) -> int:
            return 12

    fake_torch = SimpleNamespace(
        bfloat16="bf16",
        float16="fp16",
        float32="fp32",
    )
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: _Model()
        ),
    )
    fake_peft = SimpleNamespace(
        PeftModel=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: _Model()
        ),
        get_peft_model_state_dict=lambda model: {"lora.weight": _Tensor()},
    )
    monkeypatch.setattr(
        generative_runtime,
        "_import_generative_dependencies",
        lambda: (fake_torch, fake_transformers),
    )
    monkeypatch.setattr(
        generative_runtime,
        "_import_peft_dependency",
        lambda: fake_peft,
    )

    with pytest.raises(ValueError, match="parameter count mismatch"):
        TransformersCausalLMRuntime(
            model_id="Qwen/Qwen3-8B",
            revision=_REVISION,
            adapter=adapter,
        )._load()


def test_structured_response_recovers_json_after_thinking_and_fence() -> None:
    parsed = parse_structured_response(
        '<think>private reasoning</think>\n```json\n{"entities": [{"text": "ho"}]}\n```'
    )

    assert parsed == {"entities": [{"text": "ho"}]}


def test_structured_response_rejects_non_json_text() -> None:
    with pytest.raises(StructuredResponseError, match="Could not parse"):
        parse_structured_response("Không tìm thấy thực thể.")
