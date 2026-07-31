"""Joint lattice contracts: exact type labels, raw offsets, and expected-gain selection."""

from __future__ import annotations

from dataclasses import dataclass

from medical_kg_nlp.benchmarks.phase1.joint_span import (
    Phase1JointSpanCandidate,
    Phase1JointSpanLabel,
    Phase1JointSpanPrediction,
    Phase1JointSpanSelectionPolicy,
    generate_phase1_joint_span_lattice,
    label_phase1_joint_span_candidate,
    resolve_phase1_joint_span_lattice,
)
from medical_kg_nlp.benchmarks.phase1.proposal_features import ProposalSourceRole


def test_joint_lattice_selects_exact_longer_span_and_preserves_raw_offsets() -> None:
    source = "Triệu chứng hiện tại: đau ngực trái"
    lattice = generate_phase1_joint_span_lattice(
        "doc-1",
        source,
        [_row("doc-1", "đau", source.index("đau"))],
        source_roles={"xlmr": ProposalSourceRole.TOKEN_MODEL},
    )
    exact = next(candidate for candidate in lattice if candidate.variant.text == "đau ngực trái")
    short = next(candidate for candidate in lattice if candidate.variant.text == "đau")
    output, scored = resolve_phase1_joint_span_lattice(
        lattice,
        _StaticVerifier({
            exact.variant.variant_id: _distribution(exact.expected_exact_label, 0.93),
            short.variant.variant_id: _distribution(short.expected_exact_label, 0.82),
        }),
        policy=Phase1JointSpanSelectionPolicy.conservative_default(),
    )

    selected = output["doc-1"]
    assert [row["text"] for row in selected] == ["đau ngực trái"]
    start, end = selected[0]["position"]
    assert source[start:end] == selected[0]["text"]
    assert any(
        item.candidate.variant.variant_id == short.variant.variant_id
        and item.rejection_reason == "conflicts_with_higher_expected_gain"
        for item in scored
    )


def test_joint_lattice_labels_type_specific_exact_and_boundary_errors() -> None:
    source = "khó thở"
    lattice = generate_phase1_joint_span_lattice(
        "doc-2",
        source,
        [_row("doc-2", "khó", 0)],
        source_roles={"xlmr": ProposalSourceRole.RULE},
    )
    short = next(candidate for candidate in lattice if candidate.variant.text == "khó")
    exact = next(candidate for candidate in lattice if candidate.variant.text == source)
    gold = [{"text": source, "type": "TRIỆU_CHỨNG", "position": [0, len(source)]}]

    assert label_phase1_joint_span_candidate(exact, gold) is Phase1JointSpanLabel.EXACT_SYMPTOM
    assert label_phase1_joint_span_candidate(short, gold) is Phase1JointSpanLabel.TOO_SHORT


def test_joint_lattice_records_global_not_clinical_threshold_fallback() -> None:
    source = "Hỏi: đau ngực là gì?"
    lattice = generate_phase1_joint_span_lattice(
        "doc-3",
        source,
        [_row("doc-3", "đau ngực", source.index("đau"))],
        source_roles={"xlmr": ProposalSourceRole.LLM},
    )
    candidate = next(item for item in lattice if item.variant.text == "đau ngực")
    policy = Phase1JointSpanSelectionPolicy(
        tuple((entity_type, 0.6) for entity_type in ("CHẨN_ĐOÁN", "KẾT_QUẢ_XÉT_NGHIỆM", "TÊN_XÉT_NGHIỆM", "THUỐC", "TRIỆU_CHỨNG")),
        (("clinical", "TRIỆU_CHỨNG", 0.9),),
    )
    _, scored = resolve_phase1_joint_span_lattice(
        (candidate,),
        _StaticVerifier({candidate.variant.variant_id: _distribution(candidate.expected_exact_label, 0.7)}),
        policy=policy,
    )

    assert scored[0].selected is True
    assert scored[0].threshold == 0.6
    assert scored[0].threshold_source == "global_per_type"


@dataclass(frozen=True)
class _StaticVerifier:
    distributions: dict[str, tuple[tuple[str, float], ...]]

    @property
    def provenance(self) -> str:
        return "test-static-joint-verifier"

    def predict(self, candidates: list[Phase1JointSpanCandidate]):
        return tuple(
            Phase1JointSpanPrediction(candidate.variant.variant_id, self.distributions.get(candidate.variant.variant_id, _distribution(Phase1JointSpanLabel.SPURIOUS, 0.99)))
            for candidate in candidates
        )


def _distribution(label: Phase1JointSpanLabel, probability: float) -> tuple[tuple[str, float], ...]:
    remaining = (1.0 - probability) / (len(Phase1JointSpanLabel) - 1)
    return tuple(
        (candidate.value, probability if candidate is label else remaining)
        for candidate in Phase1JointSpanLabel
    )


def _row(document_id: str, text: str, start: int) -> dict[str, object]:
    return {
        "document_id": document_id,
        "proposal_id": f"{document_id}:{start}:{text}",
        "text": text,
        "type": "TRIỆU_CHỨNG",
        "position": [start, start + len(text)],
        "sources": ["xlmr"],
        "source_count": 1,
        "all_source_agreement": False,
        "status": "source_only",
        "source_evidence": {
            "xlmr": {"confidence": 0.8, "source_labels": ["TRIỆU_CHỨNG"], "support_only": False}
        },
    }
