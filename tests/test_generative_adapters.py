"""Fast contracts for structured local generative-model adapters."""

from __future__ import annotations

import pytest

from medical_kg_nlp.adapters.generative import (
    InferenceBudgetManifest,
    ModelBudgetEntry,
    StructuredResponseError,
    parse_structured_response,
)

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


def test_structured_response_recovers_json_after_thinking_and_fence() -> None:
    parsed = parse_structured_response(
        '<think>private reasoning</think>\n```json\n{"entities": [{"text": "ho"}]}\n```'
    )

    assert parsed == {"entities": [{"text": "ho"}]}


def test_structured_response_rejects_non_json_text() -> None:
    with pytest.raises(StructuredResponseError, match="Could not parse"):
        parse_structured_response("Không tìm thấy thực thể.")
