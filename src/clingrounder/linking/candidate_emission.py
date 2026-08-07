"""Metric-aware candidate-set emission with separate ICD-10 and RxNorm policies.

Retrieval and reranking estimate candidate evidence. This module owns the later decision of which
code set, including the empty set, should be emitted for a task metric. Calibration remains an
injectable artifact so benchmark-specific public probes never leak into core retrieval scores.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Literal

from clingrounder.schema.annotation import CandidateConcept
from clingrounder.schema.types import CodeSystem, EntityType

__all__ = [
    "CandidateEmissionCalibration",
    "CandidateEmissionCandidate",
    "CandidateEmissionContext",
    "CandidateEmissionDecision",
    "CandidateEmissionFeatureBucket",
    "CandidateEmissionPolicy",
    "CandidateProbabilityBucket",
    "ICDEmissionPolicy",
    "RxNormEmissionPolicy",
    "expected_jaccard_for_subset",
    "select_candidate_emission",
]


@dataclass(frozen=True, slots=True)
class CandidateEmissionCandidate:
    """One bounded candidate with calibrated and source-specific evidence."""

    code: str
    code_system: CodeSystem
    probability: float
    source: str
    exact_match: bool = False
    memory_support: int = 0
    learned_edit_precision: float | None = None
    dense_score: float | None = None
    listwise_agreement: float | None = None
    rxnorm_hard_conflict: str | None = None
    rxnorm_tty: str | None = None
    icd_is_leaf: bool | None = None

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.source.strip():
            raise ValueError("Emission candidate requires code and source")
        _probability(self.probability, "probability")
        if self.memory_support < 0:
            raise ValueError("memory_support cannot be negative")
        for field_name, value in (
            ("learned_edit_precision", self.learned_edit_precision),
            ("dense_score", self.dense_score),
            ("listwise_agreement", self.listwise_agreement),
        ):
            if value is not None:
                _probability(value, field_name)

    @classmethod
    def from_schema(
        cls,
        candidate: CandidateConcept,
        **evidence: object,
    ) -> "CandidateEmissionCandidate":
        """Adapt a qualified pipeline candidate without guessing missing calibration features."""

        if candidate.code is None:
            raise ValueError("Cannot emit a candidate without a code")
        return cls(
            code=candidate.code,
            code_system=candidate.code_system,
            probability=candidate.emit_probability,
            source=candidate.source,
            exact_match=candidate.source in {"exact", "reviewed_memory", "mention_memory"},
            memory_support=_nonnegative_integer(
                evidence.get("memory_support", 0), "memory_support"
            ),
            learned_edit_precision=_optional_float(
                evidence.get("learned_edit_precision"), "learned_edit_precision"
            ),
            dense_score=_optional_float(evidence.get("dense_score"), "dense_score"),
            listwise_agreement=_optional_float(
                evidence.get("listwise_agreement"), "listwise_agreement"
            ),
            rxnorm_hard_conflict=_optional_text(evidence.get("rxnorm_hard_conflict")),
            rxnorm_tty=_optional_text(evidence.get("rxnorm_tty")),
            icd_is_leaf=_optional_bool(evidence.get("icd_is_leaf"), "icd_is_leaf"),
        )


@dataclass(frozen=True, slots=True)
class CandidateEmissionContext:
    """Mention-level evidence shared by all candidate subsets."""

    entity_type: EntityType
    mention_seen: bool
    has_structured_evidence: bool = False
    existing_codes: tuple[str, ...] = ()
    isolated_probe: bool = False

    def __post_init__(self) -> None:
        if any(not code.strip() for code in self.existing_codes):
            raise ValueError("Existing candidate codes must be non-empty")
        if len(set(self.existing_codes)) != len(self.existing_codes):
            raise ValueError("Existing candidate codes must be unique")


@dataclass(frozen=True, slots=True)
class CandidateEmissionFeatureBucket:
    """Discrete calibration key plus diagnostic continuous features."""

    code_system: CodeSystem
    entity_type: EntityType
    primary_source: str
    mention_seen: bool
    rxnorm_tty: str | None
    has_structured_evidence: bool
    exact_match: bool
    memory_support: int
    learned_edit_precision: float | None
    dense_score: float | None
    listwise_agreement: float | None
    icd_is_leaf: bool | None
    top1_probability: float
    top1_top2_margin: float
    candidate_entropy: float

    @property
    def key(self) -> str:
        """Return a stable, deliberately coarse feature bucket identifier."""

        return "|".join(
            (
                self.code_system.value,
                self.entity_type.value,
                self.primary_source,
                f"seen={int(self.mention_seen)}",
                f"tty={self.rxnorm_tty or '*'}",
                f"structured={int(self.has_structured_evidence)}",
                f"exact={int(self.exact_match)}",
                f"memory={_count_bucket(self.memory_support)}",
                f"edit={_probability_bucket(self.learned_edit_precision)}",
                f"dense={_probability_bucket(self.dense_score)}",
                f"listwise={_probability_bucket(self.listwise_agreement)}",
                f"leaf={_optional_bool_bucket(self.icd_is_leaf)}",
                f"margin={_probability_bucket(self.top1_top2_margin)}",
                f"entropy={_probability_bucket(self.candidate_entropy)}",
            )
        )


@dataclass(frozen=True, slots=True)
class CandidateProbabilityBucket:
    """Local rank/null calibration for one feature bucket."""

    feature_key: str
    empty_probability: float
    probabilities_by_rank: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.feature_key.strip() or not self.probabilities_by_rank:
            raise ValueError("Candidate calibration bucket requires key and rank probabilities")
        _probability(self.empty_probability, "empty_probability")
        for index, value in enumerate(self.probabilities_by_rank):
            _probability(value, f"probabilities_by_rank[{index}]")


@dataclass(frozen=True, slots=True)
class CandidateEmissionCalibration:
    """Immutable local calibration artifact keyed by feature bucket."""

    buckets: tuple[CandidateProbabilityBucket, ...]

    def __post_init__(self) -> None:
        keys = [bucket.feature_key for bucket in self.buckets]
        if len(keys) != len(set(keys)):
            raise ValueError("Candidate emission calibration contains duplicate feature keys")

    def lookup(self, feature_key: str) -> CandidateProbabilityBucket | None:
        return next(
            (bucket for bucket in self.buckets if bucket.feature_key == feature_key),
            None,
        )


@dataclass(frozen=True, slots=True)
class ICDEmissionPolicy:
    """ICD policy favors useful bounded recall and preserves prior multi-code outputs."""

    preserve_existing_multicode: bool = True
    allow_compression: bool = False
    maximum_candidates: int = 5
    change_only_by_isolated_probe: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.maximum_candidates <= 5:
            raise ValueError("ICD maximum_candidates must be between 1 and 5")


@dataclass(frozen=True, slots=True)
class RxNormEmissionPolicy:
    """RxNorm policy remains conservative around structured product ambiguity."""

    maximum_candidates: int = 2
    unique_high_confidence: Literal["emit_one"] = "emit_one"
    high_confidence_threshold: float = 0.95
    low_margin: Literal["abstain"] = "abstain"
    low_margin_threshold: float = 0.05
    explicit_structure_conflict: Literal["abstain"] = "abstain"

    def __post_init__(self) -> None:
        if not 1 <= self.maximum_candidates <= 2:
            raise ValueError("RxNorm maximum_candidates must be 1 or 2")
        _probability(self.high_confidence_threshold, "high_confidence_threshold")
        _probability(self.low_margin_threshold, "low_margin_threshold")


@dataclass(frozen=True, slots=True)
class CandidateEmissionPolicy:
    """Code-system-specific emission policy."""

    icd: ICDEmissionPolicy = field(default_factory=ICDEmissionPolicy)
    rxnorm: RxNormEmissionPolicy = field(default_factory=RxNormEmissionPolicy)
    minimum_expected_jaccard_gain: float = 0.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.minimum_expected_jaccard_gain)
            or self.minimum_expected_jaccard_gain < 0.0
        ):
            raise ValueError("minimum_expected_jaccard_gain must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class CandidateEmissionDecision:
    """Selected code set and full local utility trace."""

    selected_codes: tuple[str, ...]
    expected_jaccard: float
    empty_probability: float
    feature_bucket: CandidateEmissionFeatureBucket | None
    evaluated_subsets: tuple[tuple[tuple[str, ...], float], ...]
    reason: str


def select_candidate_emission(
    candidates: tuple[CandidateEmissionCandidate, ...],
    context: CandidateEmissionContext,
    *,
    policy: CandidateEmissionPolicy | None = None,
    calibration: CandidateEmissionCalibration | None = None,
) -> CandidateEmissionDecision:
    """Enumerate valid subsets and select the locally highest expected Jaccard utility."""

    active_policy = policy or CandidateEmissionPolicy()
    if not candidates:
        return CandidateEmissionDecision((), 1.0, 1.0, None, (), "no_candidates")
    systems = {candidate.code_system for candidate in candidates}
    if len(systems) != 1:
        raise ValueError("Candidate emission requires one code system per decision")
    code_system = next(iter(systems))
    if code_system not in {CodeSystem.ICD10, CodeSystem.RXNORM}:
        raise ValueError(f"Unsupported candidate emission code system {code_system.value}")
    ordered = _deduplicate_candidates(candidates)[:5]
    feature_bucket = _feature_bucket(ordered, context)
    calibrated = calibration.lookup(feature_bucket.key) if calibration is not None else None
    probabilities = (
        tuple(calibrated.probabilities_by_rank[: len(ordered)])
        if calibrated is not None
        else tuple(candidate.probability for candidate in ordered)
    )
    if len(probabilities) != len(ordered):
        return CandidateEmissionDecision(
            (),
            calibrated.empty_probability if calibrated is not None else 1.0,
            calibrated.empty_probability if calibrated is not None else 1.0,
            feature_bucket,
            (),
            "insufficient_rank_calibration",
        )
    empty_probability = calibrated.empty_probability if calibrated is not None else _null_probability(
        probabilities
    )

    if code_system is CodeSystem.ICD10:
        return _select_icd(
            ordered,
            probabilities,
            empty_probability,
            feature_bucket,
            context,
            active_policy,
        )
    return _select_rxnorm(
        ordered,
        probabilities,
        empty_probability,
        feature_bucket,
        active_policy,
    )


def expected_jaccard_for_subset(
    probabilities: tuple[float, ...],
    selected_indices: tuple[int, ...],
) -> float:
    """Compute expected set Jaccard for an arbitrary candidate subset."""

    for index, value in enumerate(probabilities):
        _probability(value, f"probabilities[{index}]")
    if len(set(selected_indices)) != len(selected_indices) or any(
        index < 0 or index >= len(probabilities) for index in selected_indices
    ):
        raise ValueError("Selected candidate indices are invalid")
    if not selected_indices:
        return math.prod(1.0 - value for value in probabilities)
    selected = set(selected_indices)
    selected_distribution = _bernoulli_count_distribution(
        tuple(value for index, value in enumerate(probabilities) if index in selected)
    )
    omitted_distribution = _bernoulli_count_distribution(
        tuple(value for index, value in enumerate(probabilities) if index not in selected)
    )
    expected = 0.0
    for true_positive, true_mass in enumerate(selected_distribution):
        for false_negative, false_mass in enumerate(omitted_distribution):
            expected += (
                true_mass
                * false_mass
                * true_positive
                / (len(selected_indices) + false_negative)
            )
    return expected


def _select_icd(
    candidates: tuple[CandidateEmissionCandidate, ...],
    probabilities: tuple[float, ...],
    empty_probability: float,
    feature_bucket: CandidateEmissionFeatureBucket,
    context: CandidateEmissionContext,
    policy: CandidateEmissionPolicy,
) -> CandidateEmissionDecision:
    existing = context.existing_codes
    if len(existing) > policy.icd.maximum_candidates:
        raise ValueError("Existing ICD candidate set exceeds configured maximum")
    if policy.icd.preserve_existing_multicode and len(existing) > 1:
        return CandidateEmissionDecision(
            existing,
            0.0,
            empty_probability,
            feature_bucket,
            (),
            "preserve_existing_multicode",
        )
    if policy.icd.change_only_by_isolated_probe and not context.isolated_probe:
        return CandidateEmissionDecision(
            existing,
            0.0,
            empty_probability,
            feature_bucket,
            (),
            "icd_change_requires_isolated_probe",
        )
    required = set(existing) if not policy.icd.allow_compression else set()
    return _select_by_expected_jaccard(
        candidates,
        probabilities,
        empty_probability,
        feature_bucket,
        maximum_candidates=policy.icd.maximum_candidates,
        minimum_gain=policy.minimum_expected_jaccard_gain,
        required_codes=required,
        reason="icd_expected_jaccard_subset",
    )


def _select_rxnorm(
    candidates: tuple[CandidateEmissionCandidate, ...],
    probabilities: tuple[float, ...],
    empty_probability: float,
    feature_bucket: CandidateEmissionFeatureBucket,
    policy: CandidateEmissionPolicy,
) -> CandidateEmissionDecision:
    safe = tuple(
        (candidate, probability)
        for candidate, probability in zip(candidates, probabilities, strict=True)
        if candidate.rxnorm_hard_conflict is None
    )
    if not safe:
        return CandidateEmissionDecision(
            (),
            empty_probability,
            empty_probability,
            feature_bucket,
            (),
            "rxnorm_structure_conflict",
        )
    safe_candidates = tuple(item[0] for item in safe)
    safe_probabilities = tuple(item[1] for item in safe)
    margin = safe_probabilities[0] - (
        safe_probabilities[1] if len(safe_probabilities) > 1 else 0.0
    )
    if (
        safe_probabilities[0] >= policy.rxnorm.high_confidence_threshold
        and margin >= policy.rxnorm.low_margin_threshold
    ):
        code = safe_candidates[0].code
        return CandidateEmissionDecision(
            (code,),
            expected_jaccard_for_subset(safe_probabilities, (0,)),
            empty_probability,
            feature_bucket,
            (((code,), expected_jaccard_for_subset(safe_probabilities, (0,))),),
            "rxnorm_unique_high_confidence",
        )
    if len(safe_probabilities) > 1 and margin < policy.rxnorm.low_margin_threshold:
        return CandidateEmissionDecision(
            (),
            empty_probability,
            empty_probability,
            feature_bucket,
            (),
            "rxnorm_low_margin_abstain",
        )
    return _select_by_expected_jaccard(
        safe_candidates,
        safe_probabilities,
        empty_probability,
        feature_bucket,
        maximum_candidates=policy.rxnorm.maximum_candidates,
        minimum_gain=policy.minimum_expected_jaccard_gain,
        required_codes=set(),
        reason="rxnorm_expected_jaccard_subset",
    )


def _select_by_expected_jaccard(
    candidates: tuple[CandidateEmissionCandidate, ...],
    probabilities: tuple[float, ...],
    empty_probability: float,
    feature_bucket: CandidateEmissionFeatureBucket,
    *,
    maximum_candidates: int,
    minimum_gain: float,
    required_codes: set[str],
    reason: str,
) -> CandidateEmissionDecision:
    evaluated: list[tuple[tuple[str, ...], float]] = [((), empty_probability)]
    candidate_indices = range(len(candidates))
    for size in range(1, min(maximum_candidates, len(candidates)) + 1):
        for indices in itertools.combinations(candidate_indices, size):
            codes = tuple(candidates[index].code for index in indices)
            if not required_codes.issubset(codes):
                continue
            evaluated.append((codes, expected_jaccard_for_subset(probabilities, indices)))
    best_codes, best_score = max(
        evaluated,
        key=lambda item: (item[1], -len(item[0]), tuple(reversed(item[0]))),
    )
    if best_codes and best_score < empty_probability + minimum_gain:
        best_codes, best_score = (), empty_probability
    return CandidateEmissionDecision(
        best_codes,
        best_score,
        empty_probability,
        feature_bucket,
        tuple(evaluated),
        reason if best_codes else f"{reason}_abstain",
    )


def _feature_bucket(
    candidates: tuple[CandidateEmissionCandidate, ...],
    context: CandidateEmissionContext,
) -> CandidateEmissionFeatureBucket:
    probabilities = tuple(candidate.probability for candidate in candidates)
    total = sum(probabilities)
    normalized = tuple(value / total for value in probabilities) if total else ()
    entropy = -sum(value * math.log2(value) for value in normalized if value > 0.0)
    maximum_entropy = math.log2(len(normalized)) if len(normalized) > 1 else 1.0
    top = candidates[0]
    return CandidateEmissionFeatureBucket(
        code_system=top.code_system,
        entity_type=context.entity_type,
        primary_source=top.source,
        mention_seen=context.mention_seen,
        rxnorm_tty=top.rxnorm_tty,
        has_structured_evidence=context.has_structured_evidence,
        exact_match=top.exact_match,
        memory_support=top.memory_support,
        learned_edit_precision=top.learned_edit_precision,
        dense_score=top.dense_score,
        listwise_agreement=top.listwise_agreement,
        icd_is_leaf=top.icd_is_leaf,
        top1_probability=probabilities[0],
        top1_top2_margin=probabilities[0] - (probabilities[1] if len(probabilities) > 1 else 0.0),
        candidate_entropy=entropy / maximum_entropy,
    )


def _deduplicate_candidates(
    candidates: tuple[CandidateEmissionCandidate, ...],
) -> tuple[CandidateEmissionCandidate, ...]:
    best: dict[str, CandidateEmissionCandidate] = {}
    for candidate in candidates:
        previous = best.get(candidate.code)
        if previous is None or candidate.probability > previous.probability:
            best[candidate.code] = candidate
    return tuple(
        sorted(best.values(), key=lambda item: (-item.probability, item.code))
    )


def _bernoulli_count_distribution(probabilities: tuple[float, ...]) -> tuple[float, ...]:
    distribution = [1.0]
    for probability in probabilities:
        updated = [0.0] * (len(distribution) + 1)
        for count, mass in enumerate(distribution):
            updated[count] += mass * (1.0 - probability)
            updated[count + 1] += mass * probability
        distribution = updated
    return tuple(distribution)


def _null_probability(probabilities: tuple[float, ...]) -> float:
    return math.prod(1.0 - probability for probability in probabilities)


def _probability(value: float, field: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be finite and within [0, 1]")


def _optional_float(value: object, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    _probability(result, field)
    return result


def _nonnegative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _optional_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_bool(value: object, field: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _count_bucket(value: int) -> str:
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    if value < 5:
        return "2-4"
    return "5+"


def _probability_bucket(value: float | None) -> str:
    if value is None:
        return "*"
    if value < 0.5:
        return "low"
    if value < 0.8:
        return "medium"
    if value < 0.95:
        return "high"
    return "very_high"


def _optional_bool_bucket(value: bool | None) -> str:
    return "*" if value is None else str(int(value))
