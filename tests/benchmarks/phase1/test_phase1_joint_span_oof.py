"""Document-grouped OOF contracts for transformer joint span calibration."""

from __future__ import annotations

import pytest

from clingrounder.benchmarks.phase1.boundary_variants import Phase1BoundaryVariant
from clingrounder.benchmarks.phase1.joint_span import (
    Phase1JointSpanCandidate,
    Phase1JointSpanLabel,
)
from clingrounder.benchmarks.phase1.joint_span_calibration import (
    Phase1JointSpanCalibrationObservation,
)
from clingrounder.benchmarks.phase1.joint_span_oof import (
    _OofExample,
    _validate_oof_coverage,
    assign_phase1_joint_span_oof_folds,
)


def test_joint_span_oof_assignment_is_deterministic_and_document_grouped() -> None:
    first = assign_phase1_joint_span_oof_folds(
        ("doc-c", "doc-a", "doc-b", "doc-d"),
        fold_count=2,
    )
    second = assign_phase1_joint_span_oof_folds(
        ("doc-d", "doc-b", "doc-a", "doc-c"),
        fold_count=2,
    )

    assert first == second
    assert set(first.values()) == {0, 1}


def test_joint_span_oof_coverage_uses_every_candidate_and_document_once() -> None:
    examples = (_example("doc-a", "a"), _example("doc-b", "b"))
    folds = assign_phase1_joint_span_oof_folds(("doc-a", "doc-b"), fold_count=2)
    observations = tuple(
        Phase1JointSpanCalibrationObservation(
            document_id=example.candidate.variant.document_id,
            variant_id=example.candidate.variant.variant_id,
            fold=f"fold-{folds[example.candidate.variant.document_id]}",
            genre=example.candidate.genre,
            entity_type=example.candidate.variant.entity_type,
            exact_probability=0.9,
            is_exact=True,
        )
        for example in examples
    )

    coverage = _validate_oof_coverage(examples, observations, folds)

    assert coverage["candidate_coverage"] == 1.0
    assert coverage["document_coverage"] == 1.0
    assert coverage["by_genre_type"] == [
        {
            "genre": "clinical",
            "type": "TRIỆU_CHỨNG",
            "expected": 2,
            "scored": 2,
            "positive": 2,
            "negative": 0,
        }
    ]


def test_joint_span_oof_coverage_uses_explicit_source_groups() -> None:
    """Child chunks may have distinct document IDs but must share their source-note fold."""

    examples = (
        _example("chunk-1", "a", group_id="source-note-1"),
        _example("chunk-2", "b", group_id="source-note-1"),
    )
    folds = assign_phase1_joint_span_oof_folds(("source-note-1", "source-note-2"), fold_count=2)
    observations = tuple(
        Phase1JointSpanCalibrationObservation(
            document_id=example.candidate.variant.document_id,
            variant_id=example.candidate.variant.variant_id,
            fold=f"fold-{folds[example.oof_group_id]}",
            genre=example.candidate.genre,
            entity_type=example.candidate.variant.entity_type,
            exact_probability=0.9,
            is_exact=True,
        )
        for example in examples
    )

    coverage = _validate_oof_coverage(examples, observations, folds)

    assert folds["source-note-1"] == int(observations[0].fold.removeprefix("fold-"))
    assert observations[0].fold == observations[1].fold
    assert coverage["document_coverage"] == 1.0


def test_joint_span_oof_coverage_rejects_a_missing_lattice_candidate() -> None:
    examples = (_example("doc-a", "a"), _example("doc-b", "b"))
    folds = assign_phase1_joint_span_oof_folds(("doc-a", "doc-b"), fold_count=2)
    observation = Phase1JointSpanCalibrationObservation(
        document_id="doc-a",
        variant_id="variant-a",
        fold=f"fold-{folds['doc-a']}",
        genre="clinical",
        entity_type="TRIỆU_CHỨNG",
        exact_probability=0.9,
        is_exact=True,
    )

    with pytest.raises(ValueError, match="coverage mismatch"):
        _validate_oof_coverage(examples, (observation,), folds)


def _example(document_id: str, suffix: str, *, group_id: str | None = None) -> _OofExample:
    candidate = Phase1JointSpanCandidate(
        variant=Phase1BoundaryVariant(
            document_id=document_id,
            variant_id=f"variant-{suffix}",
            family_id=f"family-{suffix}",
            text="đau ngực",
            entity_type="TRIỆU_CHỨNG",
            position=(0, 8),
            sources=("rule",),
            source_evidence=(("rule", 0.9, ("SYMPTOM",), False),),
            generators=("proposal",),
            foundation_spans=((0, 8),),
            status="source_only",
            all_source_agreement=False,
        ),
        genre="clinical",
        section="triệu chứng hiện tại",
        cross_encoder_text="[GENRE] clinical [ENTITY] đau ngực",
    )
    row = {
        **candidate.to_dict(),
        "label": "EXACT_SYMPTOM",
        "source_dataset": "manual_gold",
        "oof_group_id": group_id or document_id,
    }
    return _OofExample(
        row=row,
        candidate=candidate,
        label=Phase1JointSpanLabel.EXACT_SYMPTOM,
        source_dataset="manual_gold",
        oof_group_id=group_id or document_id,
    )
