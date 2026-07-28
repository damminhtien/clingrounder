"""Lazy local Transformers runtime for reusable chat-model adapters."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

from medical_kg_nlp.adapters.huggingface.runtime import OptionalModelDependencyError
from medical_kg_nlp.utils.hashing import sha256_directory

__all__ = [
    "ChatMessage",
    "GenerationConfig",
    "GenerativeModelPort",
    "LocalPeftAdapterConfig",
    "TransformersCausalLMRuntime",
]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One model-neutral chat message."""

    role: Literal["system", "user", "assistant"]
    content: str

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("Chat message content must be non-empty")


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Bounded generation controls shared by recall and adjudication passes."""

    max_new_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    enable_thinking: bool = False
    stop_on_complete_json: bool = False

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")


class GenerativeModelPort(Protocol):
    """Replaceable interface for one locally executed chat checkpoint."""

    def generate(
        self,
        messages: Sequence[ChatMessage],
        config: GenerationConfig,
    ) -> str:
        """Generate one raw response without interpreting task semantics."""


@dataclass(frozen=True, slots=True)
class LocalPeftAdapterConfig:
    """Immutable local PEFT artifact identity used by a causal-LM runtime.

    Provenance is validated by the task run spec before construction. The runtime repeats the
    byte fingerprint check immediately before loading so a transferred adapter cannot change
    between experiment validation and model initialization.
    """

    path: Path
    fingerprint: str
    parameter_count: int

    def __post_init__(self) -> None:
        if len(self.fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.fingerprint
        ):
            raise ValueError("PEFT adapter fingerprint must be a lowercase SHA-256")
        if self.parameter_count <= 0:
            raise ValueError("PEFT adapter parameter_count must be positive")

    def verify(self) -> None:
        """Verify local adapter bytes without importing Torch, Transformers, or PEFT."""

        if not self.path.is_dir():
            raise ValueError(f"PEFT adapter directory does not exist: {self.path}")
        actual = sha256_directory(self.path)
        if actual != self.fingerprint:
            raise ValueError(
                "PEFT adapter fingerprint mismatch: "
                f"expected {self.fingerprint}, got {actual}"
            )


class TransformersCausalLMRuntime:
    """Lazy ``transformers`` implementation of :class:`GenerativeModelPort`.

    Checkpoint download is an explicit setup step. The runtime defaults to local-only loading so
    evaluation never starts network traffic or silently changes model revisions.
    """

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        device: str = "cuda",
        dtype: Literal["auto", "bf16", "fp16", "fp32"] = "bf16",
        local_files_only: bool = True,
        adapter: LocalPeftAdapterConfig | None = None,
    ) -> None:
        if not model_id.strip() or not revision.strip():
            raise ValueError("model_id and revision are required")
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.dtype = dtype
        self.local_files_only = local_files_only
        self.adapter = adapter
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def generate(
        self,
        messages: Sequence[ChatMessage],
        config: GenerationConfig,
    ) -> str:
        """Run deterministic greedy decoding unless sampling is explicitly requested."""

        if not messages:
            raise ValueError("At least one chat message is required")
        torch, tokenizer, model = self._load()
        payload = [{"role": item.role, "content": item.content} for item in messages]
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": config.enable_thinking,
        }
        try:
            prompt = tokenizer.apply_chat_template(payload, **template_kwargs)
        except TypeError:
            # MODEL: older chat templates do not expose Qwen's thinking switch.
            template_kwargs.pop("enable_thinking")
            prompt = tokenizer.apply_chat_template(payload, **template_kwargs)
        encoded = tokenizer(prompt, return_tensors="pt")
        input_device = next(model.parameters()).device
        model_inputs = {
            key: value.to(input_device)
            for key, value in encoded.items()
            if hasattr(value, "to")
        }
        prompt_length = int(model_inputs["input_ids"].shape[-1])
        torch.manual_seed(config.seed)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if config.stop_on_complete_json:
            # MODEL: extraction prompts require one JSON value. Qwen adapters can otherwise
            # continue generating until max_new_tokens even after that value is complete.
            transformers = importlib.import_module("transformers")
            generation_kwargs["stopping_criteria"] = transformers.StoppingCriteriaList(
                [_CompleteJsonStoppingCriteria(tokenizer, prompt_length)]
            )
        if config.temperature > 0:
            generation_kwargs.update(
                temperature=config.temperature,
                top_p=config.top_p,
            )
        with torch.inference_mode():
            generated = model.generate(**model_inputs, **generation_kwargs)
        return str(tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True))

    def _load(self) -> tuple[Any, Any, Any]:
        if self._model is not None:
            return self._torch, self._tokenizer, self._model
        if self.adapter is not None:
            self.adapter.verify()
        torch, transformers = _import_generative_dependencies()
        dtype = {
            "auto": "auto",
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[self.dtype]
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            local_files_only=self.local_files_only,
            trust_remote_code=False,
            use_fast=True,
        )
        model_kwargs: dict[str, Any] = {
            "revision": self.revision,
            "local_files_only": self.local_files_only,
            "trust_remote_code": False,
            "dtype": dtype,
        }
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        base_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs,
        )
        model = base_model
        if self.adapter is not None:
            peft = _import_peft_dependency()
            # MODEL: load the adapter as a separate inference artifact. Never call
            # merge_and_unload(), because the run manifest budgets and fingerprints base and
            # adapter weights independently.
            model = peft.PeftModel.from_pretrained(
                base_model,
                str(self.adapter.path),
                is_trainable=False,
                local_files_only=True,
            )
            adapter_state = peft.get_peft_model_state_dict(model)
            actual_parameter_count = sum(
                int(value.numel()) for value in adapter_state.values()
            )
            if actual_parameter_count != self.adapter.parameter_count:
                raise ValueError(
                    "PEFT adapter parameter count mismatch: "
                    f"expected {self.adapter.parameter_count:,}, "
                    f"got {actual_parameter_count:,}"
                )
        if self.device != "cuda":
            model.to(self.device)
        model.eval()
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        return torch, tokenizer, model


class _CompleteJsonStoppingCriteria:
    """Stop batch-size-one decoding after the first complete JSON object or array."""

    def __init__(self, tokenizer: Any, prompt_length: int) -> None:
        self._tokenizer = tokenizer
        self._prompt_length = prompt_length
        self._last_checked_length = 0

    def __call__(
        self,
        input_ids: Any,
        scores: Any,
        **kwargs: Any,
    ) -> bool:
        del scores, kwargs
        if len(input_ids) != 1:
            # SCALING: the current runtime intentionally generates one request at a time. Failing
            # open here avoids stopping only part of a future batch.
            return False
        generated = input_ids[0][self._prompt_length :]
        generated_length = len(generated)
        if generated_length <= self._last_checked_length:
            return False
        self._last_checked_length = generated_length
        tail = str(
            self._tokenizer.decode(generated[-2:], skip_special_tokens=True)
        )
        if "}" not in tail and "]" not in tail:
            return False
        raw_response = str(
            self._tokenizer.decode(generated, skip_special_tokens=True)
        )
        return _has_complete_outer_json(raw_response)


def _has_complete_outer_json(raw_response: str) -> bool:
    """Return true only when the first outer JSON start has completed.

    The general response parser may recover a nested object while its containing array is still
    incomplete. Generation must not stop on that nested value.
    """

    starts = [
        index
        for character in ("{", "[")
        if (index := raw_response.find(character)) >= 0
    ]
    if not starts:
        return False
    candidate = raw_response[min(starts) :]
    try:
        value, _ = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError:
        return False
    return isinstance(value, (dict, list))


def _import_generative_dependencies() -> tuple[Any, Any]:
    """Import heavyweight base-model dependencies only on first inference."""

    try:
        return (
            importlib.import_module("torch"),
            importlib.import_module("transformers"),
        )
    except (ImportError, ModuleNotFoundError) as error:
        raise OptionalModelDependencyError(
            "Local generative adapters require the 'ml' extra: uv sync --extra ml"
        ) from error


def _import_peft_dependency() -> Any:
    """Import PEFT only when a run explicitly declares a local adapter."""

    try:
        return importlib.import_module("peft")
    except (ImportError, ModuleNotFoundError) as error:
        raise OptionalModelDependencyError(
            "Local PEFT adapters require the 'ml' extra: uv sync --extra ml"
        ) from error
