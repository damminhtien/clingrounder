"""Contracts for dictionary-constrained generative listwise reranking."""

from __future__ import annotations

import json
from typing import Sequence

from medical_kg_nlp.adapters.generative import (
    ChatMessage,
    GenerationConfig,
    GenerativeModelPort,
    GenerativeListwiseRerankerAdapter,
)
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.terminology.memory import InMemoryTerminologyRepository


def test_listwise_adapter_aggregates_target_identity_across_three_orders() -> None:
    runtime = _TargetCodeRuntime(target_code="2")
    adapter = _adapter(runtime)

    ranked = adapter.rerank(_candidates(), "Bệnh nhân dùng metformin.", "metformin")

    assert ranked[0].code == "2"
    assert {candidate.code for candidate in ranked} == {"1", "2", "3"}
    assert len(runtime.prompts) == 3
    candidate_orders = [
        tuple(line.split(".", 1)[1].strip().split(" |", 1)[0] for line in _candidate_lines(prompt))
        for prompt in runtime.prompts
    ]
    assert candidate_orders[0] == ("RxNorm:1", "RxNorm:2", "RxNorm:3")
    assert candidate_orders[1] == tuple(reversed(candidate_orders[0]))


def test_listwise_adapter_fails_closed_on_out_of_set_label() -> None:
    adapter = _adapter(_StaticRuntime('{"ranking":["Z","A","B"],"abstain":false}'))
    candidates = _candidates()

    ranked = adapter.rerank(candidates, "context", "metformin")

    assert ranked == candidates


def test_listwise_adapter_preserves_base_order_on_majority_abstention() -> None:
    adapter = _adapter(
        _StaticRuntime('{"ranking":["A","B","C"],"abstain":true}')
    )
    candidates = _candidates()

    ranked = adapter.rerank(candidates, "context", "metformin")

    assert ranked == candidates


def test_listwise_adapter_prompt_contains_structured_and_terminology_metadata() -> None:
    runtime = _TargetCodeRuntime(target_code="2")
    adapter = _adapter(runtime)

    adapter.rerank(
        _candidates(),
        "Bệnh nhân dùng metformin giải phóng kéo dài.",
        "metformin 500 mg extended release tablet",
    )

    prompt = runtime.prompts[0]
    assert "[STRUCTURED_MENTION]" in prompt
    assert '"release": ["extended"]' in prompt
    assert "aliases=metformin XR" in prompt
    assert "parent=RXNORM:parent" in prompt
    assert "ingredient=metformin" in prompt


def _adapter(runtime: GenerativeModelPort) -> GenerativeListwiseRerankerAdapter:
    entries = [
        ConceptEntry(
            concept_id=f"RXNORM:{code}",
            code=code,
            code_system=CodeSystem.RXNORM,
            canonical_name=f"metformin {strength} MG Oral Tablet",
            semantic_type=EntityType.DRUG,
            aliases=("metformin XR",) if code == "2" else (),
            parents=("RXNORM:parent",),
            ingredient="metformin",
            strength=f"{strength} mg",
            dose_form="tablet",
            rxnorm_tty="SCD",
        )
        for code, strength in (("1", "250"), ("2", "500"), ("3", "850"))
    ]
    return GenerativeListwiseRerankerAdapter(
        runtime,
        InMemoryTerminologyRepository(DictionaryStore(entries)),
        generation=GenerationConfig(max_new_tokens=128, stop_on_complete_json=True),
        candidate_limit=3,
        model_weight=1.0,
        structured_retries=0,
    )


def _candidates() -> list[Candidate]:
    return [
        Candidate(
            concept_id=f"RXNORM:{code}",
            code=code,
            code_system=CodeSystem.RXNORM,
            canonical_name=f"metformin {strength} MG Oral Tablet",
            semantic_type=EntityType.DRUG,
            score=score,
            source="exact",
            matched_alias="metformin",
        )
        for code, strength, score in (
            ("1", "250", 0.9),
            ("2", "500", 0.8),
            ("3", "850", 0.7),
        )
    ]


class _TargetCodeRuntime:
    def __init__(self, *, target_code: str) -> None:
        self.target_code = target_code
        self.prompts: list[str] = []

    def generate(
        self,
        messages: Sequence[ChatMessage],
        config: GenerationConfig,
    ) -> str:
        del config
        prompt = messages[-1].content
        self.prompts.append(prompt)
        lines = _candidate_lines(prompt)
        labels = [line.split(".", 1)[0] for line in lines]
        target = next(
            label
            for label, line in zip(labels, lines, strict=True)
            if f"RxNorm:{self.target_code} |" in line
        )
        ranking = [target, *(label for label in labels if label != target)]
        return json.dumps({"ranking": ranking, "abstain": False})


class _StaticRuntime:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(
        self,
        messages: Sequence[ChatMessage],
        config: GenerationConfig,
    ) -> str:
        del messages, config
        return self.response


def _candidate_lines(prompt: str) -> list[str]:
    return [
        line
        for line in prompt.splitlines()
        if len(line) > 2 and line[0].isalpha() and line[1:3] == ". "
    ]
