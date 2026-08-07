"""Local generative-model contracts with explicit provenance and parameter budgets."""

from clingrounder.adapters.generative.budget import (
    InferenceBudgetManifest,
    ModelBudgetEntry,
)
from clingrounder.adapters.generative.budget_spec import (
    BudgetReservation,
    InferenceBudgetSpec,
    ModelParameterEvidence,
    load_inference_budget_spec,
    safetensors_parameter_count,
    verify_inference_budget_spec,
)
from clingrounder.adapters.generative.runtime import (
    ChatMessage,
    GenerationConfig,
    GenerativeModelPort,
    LocalPeftAdapterConfig,
    TransformersCausalLMRuntime,
)
from clingrounder.adapters.generative.listwise_reranker import (
    GenerativeListwiseRerankerAdapter,
    LISTWISE_RERANKER_PROMPT_VERSION,
    listwise_reranker_prompt_hash,
)
from clingrounder.adapters.generative.structured import (
    StructuredResponseError,
    parse_structured_response,
)

__all__ = [
    "ChatMessage",
    "BudgetReservation",
    "GenerationConfig",
    "GenerativeListwiseRerankerAdapter",
    "GenerativeModelPort",
    "InferenceBudgetManifest",
    "InferenceBudgetSpec",
    "LocalPeftAdapterConfig",
    "LISTWISE_RERANKER_PROMPT_VERSION",
    "ModelBudgetEntry",
    "ModelParameterEvidence",
    "StructuredResponseError",
    "TransformersCausalLMRuntime",
    "load_inference_budget_spec",
    "listwise_reranker_prompt_hash",
    "parse_structured_response",
    "safetensors_parameter_count",
    "verify_inference_budget_spec",
]
