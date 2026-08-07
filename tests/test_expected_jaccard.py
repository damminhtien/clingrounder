"""Expected-Jaccard cardinality decisions independent from task exporters."""

from __future__ import annotations

import pytest

from clingrounder.linking.expected_jaccard import expected_jaccard_prefix


def test_expected_jaccard_selects_dynamic_ranked_prefix() -> None:
    decision = expected_jaccard_prefix(
        (0.9, 0.8, 0.1),
        empty_probability=0.05,
        max_candidates=5,
    )

    assert decision.selected_size == 2
    assert len(decision.scores_by_size) == 4
    assert decision.selected_score == decision.scores_by_size[2]


def test_expected_jaccard_abstains_for_calibrated_null_prevalence() -> None:
    decision = expected_jaccard_prefix(
        (0.6,),
        empty_probability=0.8,
        max_candidates=5,
    )

    assert decision.selected_size == 0
    assert decision.selected_score == 0.8


@pytest.mark.parametrize(
    ("probabilities", "empty_probability"),
    [((1.1,), 0.0), ((0.5,), -0.1)],
)
def test_expected_jaccard_rejects_invalid_probabilities(
    probabilities: tuple[float, ...],
    empty_probability: float,
) -> None:
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        expected_jaccard_prefix(
            probabilities,
            empty_probability=empty_probability,
            max_candidates=1,
        )
