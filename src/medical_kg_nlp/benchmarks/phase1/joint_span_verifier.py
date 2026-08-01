"""Phase 1 adapter that projects a generic local cross encoder onto joint span labels."""

from __future__ import annotations

from collections.abc import Sequence

from medical_kg_nlp.adapters.huggingface.config import HuggingFaceModelConfig
from medical_kg_nlp.adapters.huggingface.multiclass_text_classifier import (
    HuggingFaceMulticlassTextClassifierAdapter,
)
from medical_kg_nlp.benchmarks.phase1.joint_span_calibration import (
    CalibratedPhase1JointSpanVerifier,
    Phase1JointSpanCalibration,
)
from medical_kg_nlp.benchmarks.phase1.joint_span import (
    Phase1JointSpanCandidate,
    Phase1JointSpanLabel,
    Phase1JointSpanPrediction,
)

__all__ = ["HuggingFacePhase1JointSpanVerifier", "calibrate_phase1_joint_span_verifier"]


class HuggingFacePhase1JointSpanVerifier:
    """Score Phase 1 lattice candidates with a pinned local transformer cross encoder."""

    def __init__(self, config: HuggingFaceModelConfig) -> None:
        self._classifier = HuggingFaceMulticlassTextClassifierAdapter(
            config,
            labels=tuple(label.value for label in Phase1JointSpanLabel),
        )

    @property
    def provenance(self) -> str:
        """Expose the model revision consumed by the submission trace."""

        return self._classifier.provenance

    def predict(
        self,
        candidates: Sequence[Phase1JointSpanCandidate],
    ) -> tuple[Phase1JointSpanPrediction, ...]:
        """Score the existing joint inputs without altering candidate identity or offsets."""

        distributions = self._classifier.predict(
            tuple(candidate.cross_encoder_text for candidate in candidates)
        )
        return tuple(
            Phase1JointSpanPrediction(
                candidate.variant.variant_id,
                distribution,
            )
            for candidate, distribution in zip(candidates, distributions, strict=True)
        )


def calibrate_phase1_joint_span_verifier(
    verifier: HuggingFacePhase1JointSpanVerifier,
    calibration: Phase1JointSpanCalibration,
) -> CalibratedPhase1JointSpanVerifier:
    """Attach a pinned OOF calibration without allowing it to change source spans or types."""

    return CalibratedPhase1JointSpanVerifier(verifier, calibration)
