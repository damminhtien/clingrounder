"""OOF calibration contracts for the learned joint span/type resolver."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from medical_kg_nlp.benchmarks.phase1.joint_span import (
    Phase1JointSpanCandidate,
    Phase1JointSpanLabel,
    Phase1JointSpanPrediction,
    generate_phase1_joint_span_lattice,
)
from medical_kg_nlp.benchmarks.phase1.joint_span_calibration import (
    CalibratedPhase1JointSpanVerifier,
    Phase1JointSpanCalibration,
    Phase1JointSpanCalibrationObservation,
    fit_phase1_joint_span_calibration,
)
from medical_kg_nlp.benchmarks.phase1.proposal_features import ProposalSourceRole


def test_joint_span_calibration_fits_every_explicit_genre_type_pair() -> None:
    calibration, report = fit_phase1_joint_span_calibration(
        _observations(),
        training_family_fingerprint="a" * 64,
        fold_assignment_sha256="b" * 64,
        false_positive_cost=1.0,
    )

    assert len(calibration.points) == 15
    assert calibration.selection_policy.threshold_for(_candidate())[0] is not None
    assert report["oof_fold_count"] == 2
    assert report["decision_authority"] == "official_submission"
    assert all(group["training"]["converged"] for group in report["groups"])


def test_joint_span_calibration_rejects_missing_qa_support() -> None:
    observations = tuple(item for item in _observations() if item.genre != "qa")

    with pytest.raises(ValueError, match="qa/"):
        fit_phase1_joint_span_calibration(
            observations,
            training_family_fingerprint="a" * 64,
            fold_assignment_sha256="b" * 64,
        )


def test_calibrated_verifier_changes_only_type_compatible_exact_probability() -> None:
    calibration, _ = fit_phase1_joint_span_calibration(
        _observations(),
        training_family_fingerprint="a" * 64,
        fold_assignment_sha256="b" * 64,
    )
    candidate = _candidate()
    raw = _distribution(candidate.expected_exact_label, 0.8)
    verifier = CalibratedPhase1JointSpanVerifier(
        _StaticVerifier({candidate.variant.variant_id: raw}),
        calibration,
    )

    calibrated = verifier.predict((candidate,))[0]

    assert calibrated.variant_id == candidate.variant.variant_id
    assert sum(value for _, value in calibrated.probabilities) == pytest.approx(1.0)
    assert calibrated.probability(candidate.expected_exact_label) != pytest.approx(
        dict(raw)[candidate.expected_exact_label.value]
    )
    assert "+calibration:" in verifier.provenance


def test_calibration_artifact_cannot_restore_a_global_fallback() -> None:
    calibration, _ = fit_phase1_joint_span_calibration(
        _observations(),
        training_family_fingerprint="a" * 64,
        fold_assignment_sha256="b" * 64,
    )
    payload = calibration.to_dict()
    payload["points"] = payload["points"][:-1]

    with pytest.raises(ValueError, match="lacks OOF support"):
        Phase1JointSpanCalibration.from_dict(payload)


def _observations() -> tuple[Phase1JointSpanCalibrationObservation, ...]:
    return tuple(
        Phase1JointSpanCalibrationObservation(
            document_id=f"{genre}-{entity_type}-{fold}-{label}",
            variant_id=f"{genre}-{entity_type}-{fold}-{label}-candidate",
            fold=fold,
            genre=genre,
            entity_type=entity_type,
            exact_probability=0.85 if label else 0.15,
            is_exact=label,
        )
        for genre in ("clinical", "educational", "qa")
        for entity_type in (
            "CHẨN_ĐOÁN",
            "THUỐC",
            "TRIỆU_CHỨNG",
            "TÊN_XÉT_NGHIỆM",
            "KẾT_QUẢ_XÉT_NGHIỆM",
        )
        for fold, label in (("fold-0", True), ("fold-1", False))
    )


def _candidate() -> Phase1JointSpanCandidate:
    source = "Hỏi: đau ngực là gì?"
    return next(
        item
        for item in generate_phase1_joint_span_lattice(
            "document",
            source,
            [_row(source)],
            source_roles={"qwen": ProposalSourceRole.LLM},
        )
        if item.variant.text == "đau ngực"
    )


def _row(source: str) -> dict[str, object]:
    start = source.index("đau ngực")
    return {
        "document_id": "document",
        "proposal_id": "document:span",
        "text": "đau ngực",
        "type": "TRIỆU_CHỨNG",
        "position": [start, start + len("đau ngực")],
        "sources": ["qwen"],
        "source_count": 1,
        "all_source_agreement": False,
        "status": "source_only",
        "source_evidence": {
            "qwen": {"confidence": 0.8, "source_labels": ["TRIỆU_CHỨNG"], "support_only": False}
        },
    }


@dataclass(frozen=True)
class _StaticVerifier:
    distributions: dict[str, tuple[tuple[str, float], ...]]

    @property
    def provenance(self) -> str:
        return "test-joint-verifier"

    def predict(self, candidates: tuple[Phase1JointSpanCandidate, ...]):
        return tuple(
            Phase1JointSpanPrediction(candidate.variant.variant_id, self.distributions[candidate.variant.variant_id])
            for candidate in candidates
        )


def _distribution(
    exact_label: Phase1JointSpanLabel,
    probability: float,
) -> tuple[tuple[str, float], ...]:
    remaining = (1.0 - probability) / (len(Phase1JointSpanLabel) - 1)
    return tuple(
        (label.value, probability if label is exact_label else remaining)
        for label in Phase1JointSpanLabel
    )
