"""Tests for code-system-specific candidate subset emission."""

from __future__ import annotations

from medical_kg_nlp.linking.candidate_emission import (
    CandidateEmissionCandidate,
    CandidateEmissionContext,
    CandidateEmissionPolicy,
    ICDEmissionPolicy,
    RxNormEmissionPolicy,
    expected_jaccard_for_subset,
    select_candidate_emission,
)
from medical_kg_nlp.schema.types import CodeSystem, EntityType


def test_icd_preserves_existing_multicode_without_compression() -> None:
    decision = select_candidate_emission(
        (_candidate("I10", 0.99), _candidate("I11", 0.2)),
        CandidateEmissionContext(
            EntityType.DISEASE,
            mention_seen=True,
            existing_codes=("I10", "I11"),
        ),
    )

    assert decision.selected_codes == ("I10", "I11")
    assert decision.reason == "preserve_existing_multicode"


def test_icd_change_requires_isolated_probe() -> None:
    decision = select_candidate_emission(
        (_candidate("I10", 0.99),),
        CandidateEmissionContext(EntityType.DISEASE, mention_seen=False),
    )

    assert decision.selected_codes == ()
    assert decision.reason == "icd_change_requires_isolated_probe"


def test_icd_subset_search_can_emit_multiple_codes() -> None:
    policy = CandidateEmissionPolicy(
        icd=ICDEmissionPolicy(change_only_by_isolated_probe=True)
    )
    decision = select_candidate_emission(
        (
            _candidate("I10", 0.9),
            _candidate("I11", 0.8),
            _candidate("I12", 0.1),
        ),
        CandidateEmissionContext(
            EntityType.DISEASE,
            mention_seen=False,
            isolated_probe=True,
        ),
        policy=policy,
    )

    assert decision.selected_codes == ("I10", "I11")
    assert len(decision.evaluated_subsets) == 8


def test_rxnorm_high_confidence_emits_one() -> None:
    decision = select_candidate_emission(
        (
            _candidate("1", 0.97, system=CodeSystem.RXNORM, tty="SCD"),
            _candidate("2", 0.50, system=CodeSystem.RXNORM, tty="SCD"),
        ),
        CandidateEmissionContext(
            EntityType.DRUG,
            mention_seen=True,
            has_structured_evidence=True,
        ),
    )

    assert decision.selected_codes == ("1",)
    assert decision.reason == "rxnorm_unique_high_confidence"


def test_rxnorm_low_margin_abstains() -> None:
    decision = select_candidate_emission(
        (
            _candidate("1", 0.70, system=CodeSystem.RXNORM),
            _candidate("2", 0.68, system=CodeSystem.RXNORM),
        ),
        CandidateEmissionContext(EntityType.DRUG, mention_seen=False),
        policy=CandidateEmissionPolicy(
            rxnorm=RxNormEmissionPolicy(low_margin_threshold=0.05)
        ),
    )

    assert decision.selected_codes == ()
    assert decision.reason == "rxnorm_low_margin_abstain"


def test_rxnorm_explicit_structure_conflicts_are_removed() -> None:
    decision = select_candidate_emission(
        (
            _candidate(
                "1",
                0.99,
                system=CodeSystem.RXNORM,
                conflict="rxnorm_product_strength_mismatch",
            ),
        ),
        CandidateEmissionContext(EntityType.DRUG, mention_seen=True),
    )

    assert decision.selected_codes == ()
    assert decision.reason == "rxnorm_structure_conflict"


def test_arbitrary_subset_expected_jaccard_is_not_prefix_only() -> None:
    first_only = expected_jaccard_for_subset((0.9, 0.1, 0.8), (0,))
    first_and_third = expected_jaccard_for_subset((0.9, 0.1, 0.8), (0, 2))

    assert first_and_third > first_only


def _candidate(
    code: str,
    probability: float,
    *,
    system: CodeSystem = CodeSystem.ICD10,
    tty: str | None = None,
    conflict: str | None = None,
) -> CandidateEmissionCandidate:
    return CandidateEmissionCandidate(
        code=code,
        code_system=system,
        probability=probability,
        source="exact",
        exact_match=True,
        rxnorm_tty=tty,
        rxnorm_hard_conflict=conflict,
    )
