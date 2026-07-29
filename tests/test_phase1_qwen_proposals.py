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
    parse_phase1_quoted_response,
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
            '{"entities":[{"text":"ho","type":"TRIỆU_CHỨNG"}]}',
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


def test_qwen_recovers_complete_rows_from_truncated_entity_array() -> None:
    runtime = _FakeRuntime(
        (
            (
                '{"entities":['
                '{"text":"ho","type":"TRIỆU_CHỨNG"},'
                '{"text":"sốt","type":"TRIỆU_CHỨNG"},'
                '{"text":"incomplete"'
            ),
        )
    )

    result = Phase1QwenAdapter(runtime).extract(
        "Bệnh nhân ho và sốt.",
        pass_id="recall",
        target_types=("TRIỆU_CHỨNG",),
        generation=GenerationConfig(),
    )

    assert len(runtime.calls) == 1
    assert [(row.span, row.entity_type) for row in result.proposals] == [
        ((10, 12), EntityType.SYMPTOM),
        ((16, 19), EntityType.SYMPTOM),
    ]
    assert result.rejected == (
        {
            "reason": "partial_entity_array_recovered",
            "recovered_count": 2,
            "pass_id": "recall",
            "window_index": 0,
        },
    )


def test_stored_qwen_response_uses_the_same_partial_recovery_policy() -> None:
    proposals, rejected = parse_phase1_quoted_response(
        ('{"entities":[{"text":"ho","type":"TRIỆU_CHỨNG"},{"text":"unfinished"')
    )

    assert [(row.text, row.entity_type) for row in proposals] == [("ho", "TRIỆU_CHỨNG")]
    assert rejected == [
        {
            "reason": "partial_entity_array_recovered",
            "recovered_count": 1,
        }
    ]


def test_qwen_does_not_recover_truncated_array_without_valid_rows() -> None:
    runtime = _FakeRuntime(
        (
            '{"entities":[{"text":"unfinished"',
            '{"entities":[]}',
        )
    )

    result = Phase1QwenAdapter(runtime).extract(
        "Bệnh nhân ho.",
        pass_id="recall",
        target_types=("TRIỆU_CHỨNG",),
        generation=GenerationConfig(),
    )

    assert len(runtime.calls) == 2
    assert not result.proposals
    assert not result.rejected


def test_qwen_exhausted_structured_response_isolated_to_one_window() -> None:
    runtime = _FakeRuntime(('{"unexpected":[]}', '{"also_unexpected":[]}'))

    result = Phase1QwenAdapter(runtime, structured_retries=1).extract(
        "Bệnh nhân ho.",
        pass_id="recall",
        target_types=("TRIỆU_CHỨNG",),
        generation=GenerationConfig(),
    )

    assert not result.proposals
    assert result.response_sha256 == ()
    assert result.raw_responses == ()
    assert result.rejected == (
        {
            "reason": "structured_response_exhausted",
            "pass_id": "recall",
            "window_index": 0,
            "error": (
                "Structured generation failed after retries: "
                "Extraction response requires exactly one recognized entity array"
            ),
        },
    )


def test_qwen_recall_rejects_fields_beyond_exact_quote_and_type() -> None:
    runtime = _FakeRuntime(
        (
            json.dumps(
                {
                    "entities": [
                        {
                            "text": "ho",
                            "type": "TRIỆU_CHỨNG",
                            "confidence": 0.0,
                        }
                    ]
                },
                ensure_ascii=False,
            ),
        )
    )
    adapter = Phase1QwenAdapter(runtime)

    result = adapter.extract(
        "ho rồi ho",
        pass_id="recall",
        target_types=("TRIỆU_CHỨNG",),
        generation=GenerationConfig(),
    )

    assert not result.proposals
    assert result.rejected == (
        {
            "proposal_index": 0,
            "reason": "invalid_proposal:unexpected_fields:confidence",
            "pass_id": "recall",
            "window_index": 0,
        },
    )
    prompt = runtime.calls[0][1].content
    assert '"confidence"' not in prompt
    assert '"left_context"' not in prompt


def test_missing_reviewer_retries_json_with_wrong_root_schema() -> None:
    runtime = _FakeRuntime(('{"unexpected":[]}', '{"entities":[]}'))

    result = Phase1QwenAdapter(runtime).review_missing(
        "Bệnh nhân ho.",
        (),
        generation=GenerationConfig(),
        max_rounds=1,
    )

    assert len(runtime.calls) == 2
    assert not result.proposals
    assert "requires exactly one recognized entity array" in runtime.calls[1][-1].content


def test_qwen_parser_accepts_bare_array_and_allowlisted_wrapper() -> None:
    responses = (
        '[{"text":"ho","type":"TRIỆU_CHỨNG"}]',
        '{"missing_entities":[]}',
    )
    runtime = _FakeRuntime(responses)
    adapter = Phase1QwenAdapter(runtime)

    first = adapter.review_missing(
        "Bệnh nhân ho.",
        (),
        generation=GenerationConfig(),
        max_rounds=1,
    )
    second = adapter.review_missing(
        "Bệnh nhân ho.",
        (),
        generation=GenerationConfig(),
        max_rounds=1,
    )

    assert [proposal.span for proposal in first.proposals] == [(10, 12)]
    assert not second.proposals


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
                        },
                        {
                            "text": "sốt",
                            "type": "TRIỆU_CHỨNG",
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
            "qwen.recall": (_proposal("qwen.recall", (4, 12), EntityType.DISEASE, 0.88),),
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
                action="REPLACE_BOUNDARY",
                replacement_text="đau đầu nhẹ",
            ),
        ),
    )
    rejected = apply_phase1_adjudication(
        text,
        (candidate,),
        (
            Phase1AdjudicationDecision(
                proposal_id="p1",
                action="REPLACE_BOUNDARY",
                replacement_text="đau bụng",
            ),
        ),
    )

    assert accepted[0].span == (0, 11)
    assert rejected == ()
    assert text[accepted[0].span[0] : accepted[0].span[1]] == "đau đầu nhẹ"


def test_qwen_verifier_returns_only_four_way_structured_actions() -> None:
    runtime = _FakeRuntime(
        (
            json.dumps(
                {
                    "decisions": [
                        {
                            "proposal_id": "p1",
                            "action": "REPLACE_TYPE",
                            "replacement_text": None,
                            "replacement_type": "CHẨN_ĐOÁN",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
        )
    )
    adapter = Phase1QwenAdapter(runtime)
    candidate = Phase1AdjudicationCandidate(
        proposal_id="p1",
        text="thiếu máu",
        entity_type="KẾT_QUẢ_XÉT_NGHIỆM",
        span=(0, 9),
        sources=("qwen.recall", "xlmr"),
        confidence=0.7,
    )

    decisions, response_sha256 = adapter.adjudicate(
        "thiếu máu",
        (candidate,),
        generation=GenerationConfig(),
    )
    projected = apply_phase1_adjudication("thiếu máu", (candidate,), decisions)

    assert response_sha256
    assert decisions == (
        Phase1AdjudicationDecision(
            proposal_id="p1",
            action="REPLACE_TYPE",
            replacement_type="CHẨN_ĐOÁN",
        ),
    )
    assert projected[0].span == candidate.span
    assert projected[0].entity_type == EntityType.DISEASE
    prompt = runtime.calls[0][1].content
    assert "KEEP|DROP|REPLACE_BOUNDARY|REPLACE_TYPE" in prompt
    assert '"confidence"' not in prompt
    assert '"evidence_quote"' not in prompt


def test_quote_projection_keeps_all_repeated_raw_occurrences() -> None:
    text = "ho nhẹ; sau đó ho tăng"
    rows, rejected = project_phase1_quoted_proposals(
        text,
        (
            Phase1QuotedProposal(
                text="ho",
                entity_type="TRIỆU_CHỨNG",
            ),
        ),
        source="qwen.recall",
        evidence_id="recall.window-0",
    )

    assert [row.span for row in rows] == [(0, 2), (15, 17)]
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
