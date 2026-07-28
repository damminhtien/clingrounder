"""Isolation tests for applying calibrated Round 2 proposal decisions."""

from __future__ import annotations

from medical_kg_nlp.benchmarks.phase1.proposal_calibration import (
    ScoredPhase1Proposal,
)
from medical_kg_nlp.benchmarks.phase1.round2_proposal_verifier import (
    apply_verified_proposal_additions,
)


def test_verified_additions_preserve_baseline_and_block_overlap() -> None:
    baseline_row = _row("đau ngực", 0, "TRIỆU_CHỨNG")
    baseline_row["assertions"] = ["isHistorical"]
    base = {"1": [baseline_row]}
    scored = [
        _scored(_proposal("ho", 12, "TRIỆU_CHỨNG", "add"), selected=True),
        _scored(_proposal("ngực", 4, "TRIỆU_CHỨNG", "overlap"), selected=True),
        _scored(_proposal("sốt", 15, "TRIỆU_CHỨNG", "low"), selected=False),
    ]

    output, decisions, counters = apply_verified_proposal_additions(base, scored)

    assert output["1"][0] == baseline_row
    assert output["1"][0] is not baseline_row
    assert output["1"][1] == {
        "text": "ho",
        "type": "TRIỆU_CHỨNG",
        "assertions": [],
        "candidates": [],
        "position": [12, 14],
    }
    assert [decision["action"] for decision in decisions] == [
        "added",
        "blocked_baseline_overlap",
        "blocked_below_threshold",
    ]
    assert counters["added"] == 1
    assert counters["output_entity_total"] == 2


def test_verified_additions_report_proposal_overlap() -> None:
    proposal = _proposal("đau", 0, "TRIỆU_CHỨNG", "proposal-overlap")
    scored = [
        ScoredPhase1Proposal(
            row=proposal,
            probability=0.8,
            threshold=0.5,
            selected_before_overlap=True,
            selected=False,
            rejection_reason="overlap",
        )
    ]

    output, decisions, counters = apply_verified_proposal_additions({"1": []}, scored)

    assert output == {"1": []}
    assert decisions[0]["action"] == "blocked_proposal_overlap"
    assert counters["blocked_proposal_overlap"] == 1


def test_verified_additions_block_recognized_heading() -> None:
    proposal = _proposal(
        "Cận lâm sàng",
        0,
        "TÊN_XÉT_NGHIỆM",
        "heading",
    )

    output, decisions, counters = apply_verified_proposal_additions(
        {"1": []},
        [_scored(proposal, selected=True)],
        source_text_by_document={"1": "Cận lâm sàng:\n"},
    )

    assert output == {"1": []}
    assert decisions[0]["action"] == "blocked_structural_heading"
    assert counters["blocked_structural_heading"] == 1


def test_verified_additions_report_pre_overlap_structural_block() -> None:
    proposal = _proposal(
        "Cận lâm sàng",
        0,
        "TÊN_XÉT_NGHIỆM",
        "heading",
    )
    scored = ScoredPhase1Proposal(
        row=proposal,
        probability=0.9,
        threshold=0.5,
        selected_before_overlap=False,
        selected=False,
        rejection_reason="structural_heading",
    )

    output, decisions, counters = apply_verified_proposal_additions(
        {"1": []},
        [scored],
    )

    assert output == {"1": []}
    assert decisions[0]["action"] == "blocked_structural_heading"
    assert counters["blocked_structural_heading"] == 1


def _row(text: str, start: int, entity_type: str) -> dict[str, object]:
    return {
        "text": text,
        "type": entity_type,
        "assertions": [],
        "candidates": [],
        "position": [start, start + len(text)],
    }


def _proposal(
    text: str,
    start: int,
    entity_type: str,
    proposal_id: str,
) -> dict[str, object]:
    return {
        **_row(text, start, entity_type),
        "document_id": "1",
        "proposal_id": proposal_id,
        "sources": ["candidate"],
        "status": "source_only",
    }


def _scored(
    row: dict[str, object],
    *,
    selected: bool,
) -> ScoredPhase1Proposal:
    return ScoredPhase1Proposal(
        row=row,
        probability=0.9 if selected else 0.1,
        threshold=0.5,
        selected_before_overlap=selected,
        selected=selected,
        rejection_reason=None if selected else "below_threshold",
    )
