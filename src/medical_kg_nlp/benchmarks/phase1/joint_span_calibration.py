"""Out-of-fold probability calibration for the learned Phase 1 joint span verifier.

The cross encoder predicts a distribution over exact, boundary-error, and spurious labels. This
module calibrates only the probability of the type-compatible exact label, separately for every
``genre × entity type`` operating point. It deliberately rejects missing support rather than
silently using a clinical calibration for educational or Q&A text.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.benchmarks.phase1.joint_span import (
    Phase1JointSpanCandidate,
    Phase1JointSpanPrediction,
    Phase1JointSpanSelectionPolicy,
    Phase1JointSpanVerifierPort,
)
from medical_kg_nlp.benchmarks.phase1.proposal_features import Phase1GenreBucket
from medical_kg_nlp.ontology.phase1 import PHASE1_ALLOWED_TYPES

__all__ = [
    "CalibratedPhase1JointSpanVerifier",
    "Phase1JointSpanCalibration",
    "Phase1JointSpanCalibrationObservation",
    "Phase1JointSpanCalibrationPoint",
    "fit_phase1_joint_span_calibration",
    "load_phase1_joint_span_calibration",
]

_SCHEMA_VERSION = "phase1-joint-span-calibration.v2"
_EPSILON = 1e-6
_REQUIRED_GENRES = ("clinical", "educational", "qa")


@dataclass(frozen=True, slots=True)
class Phase1JointSpanCalibrationObservation:
    """One document-grouped out-of-fold exact-span/type probability.

    ``fold`` proves that the model which generated this probability did not train on the
    associated document. A final-fit checkpoint must never be used to fit this artifact.
    """

    document_id: str
    variant_id: str
    fold: str
    genre: str
    entity_type: str
    exact_probability: float
    is_exact: bool

    def __post_init__(self) -> None:
        if not self.document_id.strip() or not self.variant_id.strip() or not self.fold.strip():
            raise ValueError("Joint span OOF observation identity is incomplete")
        Phase1GenreBucket(self.genre)
        if self.entity_type not in PHASE1_ALLOWED_TYPES:
            raise ValueError("Joint span OOF observation has an unsupported entity type")
        if not math.isfinite(self.exact_probability) or not 0.0 <= self.exact_probability <= 1.0:
            raise ValueError("Joint span OOF exact probability must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        """Serialize an auditable OOF observation without raw note text."""

        return {
            "document_id": self.document_id,
            "variant_id": self.variant_id,
            "fold": self.fold,
            "genre": self.genre,
            "type": self.entity_type,
            "exact_probability": self.exact_probability,
            "is_exact": self.is_exact,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase1JointSpanCalibrationObservation":
        """Parse one persisted OOF observation with strict scalar validation."""

        probability = payload.get("exact_probability")
        is_exact = payload.get("is_exact")
        if isinstance(probability, bool) or not isinstance(probability, int | float):
            raise ValueError("Joint span OOF exact_probability must be numeric")
        if not isinstance(is_exact, bool):
            raise ValueError("Joint span OOF is_exact must be boolean")
        return cls(
            document_id=_required_string(payload, "document_id"),
            variant_id=_required_string(payload, "variant_id"),
            fold=_required_string(payload, "fold"),
            genre=_required_string(payload, "genre"),
            entity_type=_required_string(payload, "type"),
            exact_probability=float(probability),
            is_exact=is_exact,
        )


@dataclass(frozen=True, slots=True)
class Phase1JointSpanCalibrationPoint:
    """One Platt mapping and expected-exact-gain threshold for a genre/type pair."""

    genre: str
    entity_type: str
    slope: float
    intercept: float
    threshold: float
    positive_count: int
    negative_count: int

    def __post_init__(self) -> None:
        Phase1GenreBucket(self.genre)
        if self.entity_type not in PHASE1_ALLOWED_TYPES:
            raise ValueError("Joint span calibration point has an unsupported entity type")
        if not math.isfinite(self.slope) or not math.isfinite(self.intercept):
            raise ValueError("Joint span calibration parameters must be finite")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("Joint span calibration threshold must be in (0, 1)")
        if self.positive_count < 1 or self.negative_count < 1:
            raise ValueError("Joint span calibration needs positive and negative OOF support")

    def calibrate(self, probability: float) -> float:
        """Return the Platt-calibrated exact probability for one raw softmax value."""

        return _sigmoid(self.slope * _logit(probability) + self.intercept)

    def to_dict(self) -> dict[str, Any]:
        return {
            "genre": self.genre,
            "type": self.entity_type,
            "slope": self.slope,
            "intercept": self.intercept,
            "threshold": self.threshold,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase1JointSpanCalibrationPoint":
        """Load one point while rejecting implicit numeric or support coercions."""

        return cls(
            genre=_required_string(payload, "genre"),
            entity_type=_required_string(payload, "type"),
            slope=_required_number(payload, "slope"),
            intercept=_required_number(payload, "intercept"),
            threshold=_required_number(payload, "threshold"),
            positive_count=_required_int(payload, "positive_count"),
            negative_count=_required_int(payload, "negative_count"),
        )


@dataclass(frozen=True, slots=True)
class Phase1JointSpanCalibration:
    """Pinned OOF-derived mappings for one reproducible verifier training family.

    OOF fold checkpoints necessarily differ from the final-fit checkpoint byte-for-byte. The
    training-family fingerprint instead pins their shared supervision, initializer, label space,
    and optimizer contract without falsely claiming that a final-fit model generated OOF scores.
    """

    training_family_fingerprint: str
    oof_observations_sha256: str
    fold_assignment_sha256: str
    false_positive_cost: float
    points: tuple[Phase1JointSpanCalibrationPoint, ...]

    def __post_init__(self) -> None:
        for value in (
            self.training_family_fingerprint,
            self.oof_observations_sha256,
            self.fold_assignment_sha256,
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError("Joint span calibration fingerprints must be lowercase SHA-256")
        if not math.isfinite(self.false_positive_cost) or self.false_positive_cost <= 0.0:
            raise ValueError("Joint span calibration false_positive_cost must be finite and positive")
        keys = [(point.genre, point.entity_type) for point in self.points]
        if len(keys) != len(set(keys)):
            raise ValueError("Joint span calibration points must be unique by genre/type")
        required = {(genre, entity_type) for genre in _REQUIRED_GENRES for entity_type in PHASE1_ALLOWED_TYPES}
        missing = sorted(required - set(keys))
        if missing:
            rendered = ", ".join(f"{genre}/{entity_type}" for genre, entity_type in missing)
            raise ValueError(f"Joint span calibration lacks OOF support for: {rendered}")

    @property
    def point_by_key(self) -> dict[tuple[str, str], Phase1JointSpanCalibrationPoint]:
        """Return the immutable mapping used by runtime calibration."""

        return {(point.genre, point.entity_type): point for point in self.points}

    @property
    def selection_policy(self) -> Phase1JointSpanSelectionPolicy:
        """Project the calibrated operating points into the resolver's submission policy."""

        return Phase1JointSpanSelectionPolicy(
            genre_type_thresholds=tuple(
                sorted((point.genre, point.entity_type, point.threshold) for point in self.points)
            ),
            false_positive_cost=self.false_positive_cost,
        )

    @property
    def provenance(self) -> str:
        """Return a content-derived calibration identity for model traces."""

        serialized = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def calibrate_prediction(
        self,
        candidate: Phase1JointSpanCandidate,
        prediction: Phase1JointSpanPrediction,
    ) -> Phase1JointSpanPrediction:
        """Calibrate only the candidate's compatible exact class and preserve a distribution.

        INVARIANT: the candidate identity stays untouched. Remaining class mass is scaled
        proportionally, so the boundary/spurious evidence stays available for diagnostics.
        """

        point = self.point_by_key.get((candidate.genre, candidate.variant.entity_type))
        if point is None:
            raise ValueError("Runtime candidate lacks a pinned joint calibration point")
        expected = candidate.expected_exact_label.value
        raw = dict(prediction.probabilities)
        calibrated = point.calibrate(raw[expected])
        remaining_labels = [label for label in raw if label != expected]
        remaining_mass = sum(raw[label] for label in remaining_labels)
        adjusted: dict[str, float] = {expected: calibrated}
        if remaining_mass <= _EPSILON:
            share = (1.0 - calibrated) / len(remaining_labels)
            adjusted.update({label: share for label in remaining_labels})
        else:
            scale = (1.0 - calibrated) / remaining_mass
            adjusted.update({label: raw[label] * scale for label in remaining_labels})
        return Phase1JointSpanPrediction(
            variant_id=prediction.variant_id,
            probabilities=tuple((label, adjusted[label]) for label, _ in prediction.probabilities),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the versioned portable artifact; the caller pins its file SHA separately."""

        return {
            "schema_version": _SCHEMA_VERSION,
            "training_family_fingerprint": self.training_family_fingerprint,
            "oof_observations_sha256": self.oof_observations_sha256,
            "fold_assignment_sha256": self.fold_assignment_sha256,
            "false_positive_cost": self.false_positive_cost,
            "points": [point.to_dict() for point in sorted(self.points, key=lambda item: (item.genre, item.entity_type))],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase1JointSpanCalibration":
        """Load a portable artifact without accepting an implicit global fallback."""

        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("Unsupported joint span calibration schema")
        raw_points = payload.get("points")
        cost = payload.get("false_positive_cost")
        if not isinstance(raw_points, list) or any(not isinstance(row, Mapping) for row in raw_points):
            raise ValueError("Joint span calibration points must be a list of mappings")
        if isinstance(cost, bool) or not isinstance(cost, int | float):
            raise ValueError("Joint span calibration false_positive_cost must be numeric")
        return cls(
            training_family_fingerprint=_required_string(
                payload, "training_family_fingerprint"
            ),
            oof_observations_sha256=_required_string(payload, "oof_observations_sha256"),
            fold_assignment_sha256=_required_string(payload, "fold_assignment_sha256"),
            false_positive_cost=float(cost),
            points=tuple(Phase1JointSpanCalibrationPoint.from_dict(row) for row in raw_points),
        )


class CalibratedPhase1JointSpanVerifier:
    """Apply a pinned OOF calibration after a local transformer predicts each lattice row."""

    def __init__(
        self,
        base: Phase1JointSpanVerifierPort,
        calibration: Phase1JointSpanCalibration,
    ) -> None:
        self._base = base
        self._calibration = calibration

    @property
    def provenance(self) -> str:
        """Retain both model and calibration identities in every exported trace."""

        return f"{self._base.provenance}+calibration:{self._calibration.provenance}"

    def predict(
        self,
        candidates: Sequence[Phase1JointSpanCandidate],
    ) -> tuple[Phase1JointSpanPrediction, ...]:
        """Return calibrated distributions in the original deterministic candidate order."""

        raw = tuple(self._base.predict(candidates))
        if len(raw) != len(candidates):
            raise ValueError("Joint span base verifier must return one prediction per candidate")
        return tuple(
            self._calibration.calibrate_prediction(candidate, prediction)
            for candidate, prediction in zip(candidates, raw, strict=True)
        )


def fit_phase1_joint_span_calibration(
    observations: Sequence[Phase1JointSpanCalibrationObservation],
    *,
    training_family_fingerprint: str,
    fold_assignment_sha256: str,
    false_positive_cost: float = 1.0,
) -> tuple[Phase1JointSpanCalibration, dict[str, Any]]:
    """Fit genre/type Platt mappings from strictly document-grouped OOF probabilities.

    ``TP - false_positive_cost * FP`` chooses each gate. The global resolver later accounts for
    overlap, while this local operating point rejects candidates whose expected exact-span gain is
    already negative.
    """

    if not observations:
        raise ValueError("Joint span calibration requires OOF observations")
    if not math.isfinite(false_positive_cost) or false_positive_cost <= 0.0:
        raise ValueError("Joint span calibration false_positive_cost must be finite and positive")
    _validate_oof_observations(observations)
    groups: dict[tuple[str, str], list[Phase1JointSpanCalibrationObservation]] = defaultdict(list)
    for observation in observations:
        if observation.genre in _REQUIRED_GENRES:
            groups[(observation.genre, observation.entity_type)].append(observation)

    points: list[Phase1JointSpanCalibrationPoint] = []
    report_groups: list[dict[str, Any]] = []
    for genre in _REQUIRED_GENRES:
        for entity_type in sorted(PHASE1_ALLOWED_TYPES):
            group = groups.get((genre, entity_type), [])
            positives = sum(observation.is_exact for observation in group)
            negatives = len(group) - positives
            if positives < 1 or negatives < 1:
                raise ValueError(
                    "Joint span calibration requires positive and negative OOF support for "
                    f"{genre}/{entity_type}"
                )
            slope, intercept, training = _fit_platt(group)
            calibrated = tuple(
                _sigmoid(slope * _logit(item.exact_probability) + intercept) for item in group
            )
            threshold, utility = _select_expected_gain_threshold(
                calibrated,
                tuple(item.is_exact for item in group),
                false_positive_cost=false_positive_cost,
            )
            point = Phase1JointSpanCalibrationPoint(
                genre=genre,
                entity_type=entity_type,
                slope=slope,
                intercept=intercept,
                threshold=threshold,
                positive_count=positives,
                negative_count=negatives,
            )
            points.append(point)
            report_groups.append(
                {
                    **point.to_dict(),
                    "raw_brier": _brier(tuple(item.exact_probability for item in group), tuple(item.is_exact for item in group)),
                    "calibrated_brier": _brier(calibrated, tuple(item.is_exact for item in group)),
                    "selected_expected_gain": utility,
                    "training": training,
                }
            )
    serialized = "".join(
        json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for item in sorted(observations, key=lambda item: (item.document_id, item.variant_id, item.fold))
    )
    calibration = Phase1JointSpanCalibration(
        training_family_fingerprint=training_family_fingerprint,
        oof_observations_sha256=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        fold_assignment_sha256=fold_assignment_sha256,
        false_positive_cost=false_positive_cost,
        points=tuple(points),
    )
    report = {
        "schema_version": "phase1-joint-span-calibration-report.v1",
        "decision_authority": "official_submission",
        "local_metrics_role": "diagnostic_only",
        "oof_document_count": len({item.document_id for item in observations}),
        "oof_fold_count": len({item.fold for item in observations}),
        "groups": report_groups,
        "calibration": calibration.to_dict(),
    }
    return calibration, report


def load_phase1_joint_span_calibration(path: str | Path) -> Phase1JointSpanCalibration:
    """Load one calibration file; callers verify its pinned file SHA before this step."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Joint span calibration must be a JSON mapping")
    return Phase1JointSpanCalibration.from_dict(payload)


def _validate_oof_observations(
    observations: Sequence[Phase1JointSpanCalibrationObservation],
) -> None:
    keys: set[tuple[str, str]] = set()
    documents_by_fold: dict[str, set[str]] = defaultdict(set)
    folds_by_document: dict[str, set[str]] = defaultdict(set)
    for observation in observations:
        key = (observation.document_id, observation.variant_id)
        if key in keys:
            raise ValueError("Joint span OOF observations contain duplicate candidate identities")
        keys.add(key)
        documents_by_fold[observation.fold].add(observation.document_id)
        folds_by_document[observation.document_id].add(observation.fold)
    if len(documents_by_fold) < 2:
        raise ValueError("Joint span calibration requires at least two document-grouped folds")
    if any(len(folds) != 1 for folds in folds_by_document.values()):
        raise ValueError("A joint span calibration document must belong to exactly one OOF fold")


def _fit_platt(
    observations: Sequence[Phase1JointSpanCalibrationObservation],
) -> tuple[float, float, dict[str, Any]]:
    """Fit regularized Platt scaling with a deterministic damped Newton solver.

    The old gradient loop routinely exhausted its step budget and persisted
    ``converged=false``. Calibration is a gating artifact, so an unconverged fit is unsafe: this
    solver either reaches a stationary optimum or raises instead of writing a misleading report.
    """

    slope, intercept = 1.0, 0.0
    l2 = 0.01
    tolerance = 1e-8
    maximum_steps = 100
    values = tuple((_logit(item.exact_probability), float(item.is_exact)) for item in observations)
    current_loss = _platt_loss(values, slope, intercept, l2=l2)
    for step in range(1, maximum_steps + 1):
        gradient_slope = l2 * (slope - 1.0)
        gradient_intercept = l2 * intercept
        hessian_ss = l2
        hessian_si = 0.0
        hessian_ii = l2
        for value, label in values:
            probability = _sigmoid(slope * value + intercept)
            error = probability - label
            curvature = probability * (1.0 - probability)
            gradient_slope += error * value
            gradient_intercept += error
            hessian_ss += curvature * value * value
            hessian_si += curvature * value
            hessian_ii += curvature
        scale = 1.0 / len(values)
        gradient_slope = gradient_slope * scale + l2 * (1.0 - scale) * (slope - 1.0)
        gradient_intercept = gradient_intercept * scale + l2 * (1.0 - scale) * intercept
        hessian_ss = hessian_ss * scale + l2 * (1.0 - scale)
        hessian_si *= scale
        hessian_ii = hessian_ii * scale + l2 * (1.0 - scale)
        gradient_norm = max(abs(gradient_slope), abs(gradient_intercept))
        if gradient_norm < tolerance:
            return slope, intercept, {
                "steps": step,
                "converged": True,
                "l2": l2,
                "gradient_norm": gradient_norm,
                "solver": "damped_newton",
            }
        determinant = hessian_ss * hessian_ii - hessian_si * hessian_si
        if determinant <= _EPSILON:
            raise RuntimeError("Joint span Platt Hessian is singular")
        delta_slope = (hessian_ii * gradient_slope - hessian_si * gradient_intercept) / determinant
        delta_intercept = (
            -hessian_si * gradient_slope + hessian_ss * gradient_intercept
        ) / determinant
        accepted = False
        for backtrack in range(25):
            scale_factor = 0.5**backtrack
            next_slope = max(-12.0, min(12.0, slope - scale_factor * delta_slope))
            next_intercept = max(
                -12.0,
                min(12.0, intercept - scale_factor * delta_intercept),
            )
            next_loss = _platt_loss(values, next_slope, next_intercept, l2=l2)
            if next_loss < current_loss - 1e-14:
                slope, intercept, current_loss, accepted = (
                    next_slope,
                    next_intercept,
                    next_loss,
                    True,
                )
                break
        if not accepted:
            raise RuntimeError("Joint span Platt fit could not reduce its objective")
    raise RuntimeError(
        "Joint span Platt fit did not converge; inspect OOF score separation before calibration"
    )


def _platt_loss(
    values: Sequence[tuple[float, float]],
    slope: float,
    intercept: float,
    *,
    l2: float,
) -> float:
    """Return the regularized mean logistic loss without numeric overflow."""

    loss = 0.0
    for value, label in values:
        logit = slope * value + intercept
        # ``logaddexp(0, logit) - label * logit`` is stable for extreme probabilities.
        loss += max(logit, 0.0) + math.log1p(math.exp(-abs(logit))) - label * logit
    return loss / len(values) + 0.5 * l2 * ((slope - 1.0) ** 2 + intercept**2)


def _select_expected_gain_threshold(
    probabilities: Sequence[float],
    labels: Sequence[bool],
    *,
    false_positive_cost: float,
) -> tuple[float, float]:
    """Choose the OOF threshold maximizing observed exact-span gain with stable tie-breaks."""

    candidates = sorted({max(_EPSILON, min(1.0 - _EPSILON, value)) for value in probabilities})
    utility_floor = false_positive_cost / (1.0 + false_positive_cost)
    candidates.append(utility_floor)
    best: tuple[float, float, int] | None = None
    for threshold in sorted(set(candidates)):
        true_positive = sum(label and probability >= threshold for probability, label in zip(probabilities, labels, strict=True))
        false_positive = sum((not label) and probability >= threshold for probability, label in zip(probabilities, labels, strict=True))
        utility = float(true_positive) - false_positive_cost * float(false_positive)
        # Prefer higher utility, then a lower gate for recall, then a lower numeric threshold.
        candidate = (utility, -threshold, true_positive)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return -best[1], best[0]


def _brier(probabilities: Sequence[float], labels: Sequence[bool]) -> float:
    return sum((probability - float(label)) ** 2 for probability, label in zip(probabilities, labels, strict=True)) / len(probabilities)


def _logit(probability: float) -> float:
    bounded = max(_EPSILON, min(1.0 - _EPSILON, probability))
    return math.log(bounded / (1.0 - bounded))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _required_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Joint span calibration {field} must be a non-empty string")
    return value.strip()


def _required_number(payload: Mapping[str, Any], field: str) -> float:
    """Return one finite JSON number without accepting booleans as integers."""

    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Joint span calibration {field} must be numeric")
    return float(value)


def _required_int(payload: Mapping[str, Any], field: str) -> int:
    """Return one strict JSON integer used for observed OOF support."""

    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Joint span calibration {field} must be an integer")
    return value
