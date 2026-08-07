"""Candidate-prefix selection that optimizes expected set Jaccard."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "ExpectedJaccardDecision",
    "expected_jaccard_prefix",
]


@dataclass(frozen=True, slots=True)
class ExpectedJaccardDecision:
    """Selected ranked-prefix size and all evaluated cardinality utilities."""

    selected_size: int
    selected_score: float
    empty_score: float
    scores_by_size: tuple[float, ...]


def expected_jaccard_prefix(
    probabilities: Sequence[float],
    *,
    empty_probability: float,
    max_candidates: int,
    minimum_gain: float = 0.0,
) -> ExpectedJaccardDecision:
    """Choose ``k`` that maximizes expected Jaccard for ranked candidate prefixes.

    Candidate inclusion events are modeled as independent Bernoulli variables. The null outcome
    is calibrated separately because hidden-gold empty prevalence cannot be reconstructed from
    alternative candidate marginals.
    """

    _validate_probability(empty_probability, "empty_probability")
    if max_candidates < 0:
        raise ValueError("max_candidates cannot be negative")
    if not math.isfinite(minimum_gain) or minimum_gain < 0.0:
        raise ValueError("minimum_gain must be finite and non-negative")
    active = tuple(float(value) for value in probabilities)
    for index, probability in enumerate(active):
        _validate_probability(probability, f"probabilities[{index}]")
    limit = min(max_candidates, len(active))
    scores = [empty_probability]
    for size in range(1, limit + 1):
        selected_distribution = _bernoulli_count_distribution(active[:size])
        omitted_distribution = _bernoulli_count_distribution(active[size:])
        expected = 0.0
        for true_positive, true_positive_probability in enumerate(
            selected_distribution
        ):
            for false_negative, false_negative_probability in enumerate(
                omitted_distribution
            ):
                expected += (
                    true_positive_probability
                    * false_negative_probability
                    * true_positive
                    / (size + false_negative)
                )
        scores.append(expected)
    best_size = max(range(len(scores)), key=lambda size: (scores[size], -size))
    if scores[best_size] < empty_probability + minimum_gain:
        best_size = 0
    return ExpectedJaccardDecision(
        selected_size=best_size,
        selected_score=scores[best_size],
        empty_score=empty_probability,
        scores_by_size=tuple(scores),
    )


def _bernoulli_count_distribution(
    probabilities: Sequence[float],
) -> tuple[float, ...]:
    distribution = [1.0]
    for probability in probabilities:
        updated = [0.0] * (len(distribution) + 1)
        for count, mass in enumerate(distribution):
            updated[count] += mass * (1.0 - probability)
            updated[count + 1] += mass * probability
        distribution = updated
    return tuple(distribution)


def _validate_probability(value: float, field: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be finite and within [0, 1]")
