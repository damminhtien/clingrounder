"""Contracts for Qwen Phase 1 exact-quote projection and consensus."""

from __future__ import annotations

import json
from collections.abc import Sequence

from medical_kg_nlp.adapters.generative import ChatMessage, GenerationConfig
from medical_kg_nlp.benchmarks.phase1.qwen_proposals import (
    Phase1AdjudicationCandidate,
    Phase1AdjudicationDecision,
    Phase1QwenAdapter,
    Phase1QuotedProposal,
    Phase1ReviewEntity,
    apply_phase1_adjudication,
    project_phase1_quoted_proposals,
    select_qwen_confirmed_proposals,
    split_raw_text_windows,
)
from medical_kg_nlp.ner.proposal import EntityProposal
from medical_kg_nlp.schema.types import EntityType


class _FakeRuntime:
    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = list(responses)
        self.calls: list[Sequence[ChatMessage]] = []

    def generate(
        self,
        messages: Sequence[ChatMessage],
        config: GenerationConfig,
    ) -> str:
        del config
        self.calls.append(messages)
        return self.responses.pop(0)


def test_qwen_pass_projects_repeated_quotes_and_deduplicates_chunk_overlap() -> None:
    text = ("ho kéo dài. " * 40).strip()
    response = json.dumps(
        {
            "entities": [
                {
                    "text": "ho",
                    "type": "TRIỆU_CHỨNG",
                    "left_context": "",
                    "right_context": "",
                    "confidence": 0.91,
                }
            ]
        },
        ensure_ascii=False,
    )
    windows = split_raw_text_windows(text, max_characters=256, overlap_characters=64)
    runtime = _FakeRuntime([response] * len(windows))
    adapter = Phase1QwenAdapter(
        runtime,
        max_window_characters=256,
        window_overlap_characters=64,
    )

    result = adapter.extract(
        text,
        pass_id="recall",
        target_types=("TRIỆU_CHỨNG",),
        generation=GenerationConfig(),
    )

    assert len(result.proposals) == 40
    assert all(text[start:end] == "ho" for start, end in (row.span for row in result.proposals))
    assert not result.rejected


def test_qwen_structured_retry_does_not_accept_free_text() -> None:
    runtime = _FakeRuntime(
        (
            "Tôi tìm thấy ho.",
            '{"entities":[{"text":"ho","type":"TRIỆU_CHỨNG","confidence":0.8}]}',
        )
    )

    result = Phase1QwenAdapter(runtime).extract(
        "Bệnh nhân ho.",
        pass_id="recall",
        target_types=("TRIỆU_CHỨNG",),
        generation=GenerationConfig(),
    )

    assert len(runtime.calls) == 2
    assert result.proposals[0].span == (10, 12)


def test_missing_reviewer_retries_json_with_wrong_root_schema() -> None:
    runtime = _FakeRuntime(('{"missing":[]}', '{"entities":[]}'))

    result = Phase1QwenAdapter(runtime).review_missing(
        "Bệnh nhân ho.",
        (),
        generation=GenerationConfig(),
        max_rounds=1,
    )

    assert len(runtime.calls) == 2
    assert not result.proposals
    assert 'requires an entities array' in runtime.calls[1][-1].content


def test_missing_reviewer_projects_only_unlabeled_occurrences() -> None:
    text = "ho và sốt; ho lại"
    runtime = _FakeRuntime(
        (
            json.dumps(
                {
                    "entities": [
                        {
                            "text": "ho",
                            "type": "TRIỆU_CHỨNG",
                            "confidence": 0.9,
                        },
                        {
                            "text": "sốt",
                            "type": "TRIỆU_CHỨNG",
                            "confidence": 0.9,
                        },
                    ]
                },
                ensure_ascii=False,
            ),
            '{"entities":[]}',
        )
    )

    result = Phase1QwenAdapter(runtime).review_missing(
        text,
        (
            Phase1ReviewEntity(
                text="ho",
                entity_type="TRIỆU_CHỨNG",
                span=(0, 2),
            ),
        ),
        generation=GenerationConfig(),
        max_rounds=2,
    )

    assert [(row.span, row.entity_type) for row in result.proposals] == [
        ((6, 9), EntityType.SYMPTOM),
        ((11, 13), EntityType.SYMPTOM),
    ]
    assert "EXISTING_ENTITIES" in runtime.calls[0][1].content
    assert '"position"' not in runtime.calls[0][1].content


def test_consensus_requires_qwen_and_blocks_xlmr_only() -> None:
    xlmr = _proposal("xlmr", (0, 2), EntityType.SYMPTOM, 0.99)
    qwen = _proposal("qwen.recall", (0, 2), EntityType.SYMPTOM, 0.81)
    rule = _proposal("rule", (0, 2), EntityType.SYMPTOM, 0.7)
    xlmr_only = _proposal("xlmr", (5, 12), EntityType.SYMPTOM, 0.95)

    selected = select_qwen_confirmed_proposals(
        {
            "qwen.recall": (qwen,),
            "rule": (rule,),
            "xlmr": (xlmr, xlmr_only),
        },
        thresholds={EntityType.SYMPTOM: 0.8},
    )

    assert [row.span for row in selected] == [(0, 2)]
    assert selected[0].feature("agreement_sources") == "qwen.recall,rule,xlmr"


def test_two_qwen_passes_can_confirm_without_xlmr() -> None:
    selected = select_qwen_confirmed_proposals(
        {
            "qwen.recall": (
                _proposal("qwen.recall", (4, 12), EntityType.DISEASE, 0.88),
            ),
            "qwen.targeted.disease": (
                _proposal("qwen.targeted.disease", (4, 12), EntityType.DISEASE, 0.84),
            ),
        },
        thresholds={EntityType.DISEASE: 0.8},
    )

    assert len(selected) == 1
    assert selected[0].span == (4, 12)


def test_adjudication_replacement_must_overlap_original_span() -> None:
    text = "đau đầu nhẹ; đau bụng"
    candidate = Phase1AdjudicationCandidate(
        proposal_id="p1",
        text="đau đầu",
        entity_type="TRIỆU_CHỨNG",
        span=(0, 7),
        sources=("rule", "xlmr"),
        confidence=0.7,
    )

    accepted = apply_phase1_adjudication(
        text,
        (candidate,),
        (
            Phase1AdjudicationDecision(
                proposal_id="p1",
                action="REPLACE",
                confidence=0.93,
                evidence_quote="đau đầu nhẹ",
                replacement_text="đau đầu nhẹ",
                replacement_type="TRIỆU_CHỨNG",
            ),
        ),
        minimum_confidence=0.8,
    )
    rejected = apply_phase1_adjudication(
        text,
        (candidate,),
        (
            Phase1AdjudicationDecision(
                proposal_id="p1",
                action="REPLACE",
                confidence=0.93,
                evidence_quote="đau bụng",
                replacement_text="đau bụng",
                replacement_type="TRIỆU_CHỨNG",
            ),
        ),
        minimum_confidence=0.8,
    )

    assert accepted[0].span == (0, 11)
    assert rejected == ()
    assert text[accepted[0].span[0] : accepted[0].span[1]] == "đau đầu nhẹ"


def test_quote_projection_uses_context_to_disambiguate_repeated_mentions() -> None:
    text = "ho nhẹ; sau đó ho tăng"
    rows, rejected = project_phase1_quoted_proposals(
        text,
        (
            Phase1QuotedProposal(
                text="ho",
                entity_type="TRIỆU_CHỨNG",
                confidence=0.9,
                left_context="sau đó ",
            ),
        ),
        source="qwen.recall",
        evidence_id="recall.window-0",
    )

    assert [row.span for row in rows] == [(15, 17)]
    assert rejected == []


def _proposal(
    source: str,
    span: tuple[int, int],
    entity_type: EntityType,
    score: float,
) -> EntityProposal:
    return EntityProposal(
        span=span,
        candidate_types=(entity_type,),
        source=source,
        score=score,
    )
