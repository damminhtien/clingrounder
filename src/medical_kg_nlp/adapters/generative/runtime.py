"""Lazy local Transformers runtime for reusable chat-model adapters."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence

from medical_kg_nlp.adapters.huggingface.runtime import OptionalModelDependencyError

__all__ = [
    "ChatMessage",
    "GenerationConfig",
    "GenerativeModelPort",
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
    ) -> None:
        if not model_id.strip() or not revision.strip():
            raise ValueError("model_id and revision are required")
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.dtype = dtype
        self.local_files_only = local_files_only
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
        torch.manual_seed(config.seed)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if config.temperature > 0:
            generation_kwargs.update(
                temperature=config.temperature,
                top_p=config.top_p,
            )
        with torch.inference_mode():
            generated = model.generate(**model_inputs, **generation_kwargs)
        prompt_length = int(model_inputs["input_ids"].shape[-1])
        return str(tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True))

    def _load(self) -> tuple[Any, Any, Any]:
        if self._model is not None:
            return self._torch, self._tokenizer, self._model
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except (ImportError, ModuleNotFoundError) as error:
            raise OptionalModelDependencyError(
                "Local generative adapters require the 'ml' extra: uv sync --extra ml"
            ) from error
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
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs,
        )
        if self.device != "cuda":
            model.to(self.device)
        model.eval()
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        return torch, tokenizer, model
