"""Local generative-model contracts with explicit provenance and parameter budgets."""

from medical_kg_nlp.adapters.generative.budget import (
    InferenceBudgetManifest,
    ModelBudgetEntry,
)
from medical_kg_nlp.adapters.generative.budget_spec import (
    BudgetReservation,
    InferenceBudgetSpec,
    ModelParameterEvidence,
    load_inference_budget_spec,
    safetensors_parameter_count,
    verify_inference_budget_spec,
)
from medical_kg_nlp.adapters.generative.runtime import (
    ChatMessage,
    GenerationConfig,
    GenerativeModelPort,
    LocalPeftAdapterConfig,
    TransformersCausalLMRuntime,
)
from medical_kg_nlp.adapters.generative.structured import (
    StructuredResponseError,
    parse_structured_response,
)

__all__ = [
    "ChatMessage",
    "BudgetReservation",
    "GenerationConfig",
    "GenerativeModelPort",
    "InferenceBudgetManifest",
    "InferenceBudgetSpec",
    "LocalPeftAdapterConfig",
    "ModelBudgetEntry",
    "ModelParameterEvidence",
    "StructuredResponseError",
    "TransformersCausalLMRuntime",
    "load_inference_budget_spec",
    "parse_structured_response",
    "safetensors_parameter_count",
    "verify_inference_budget_spec",
]
