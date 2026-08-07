from clingrounder.evaluation.linking_metrics import coverage_accuracy_curve
from clingrounder.schema.annotation import CandidateConcept, EntityAnnotation
from clingrounder.schema.types import CodeSystem, EntityType


def test_linking_coverage_accuracy_curve_tracks_selective_prediction() -> None:
    gold = [_entity("E1", (0, 1), "I10"), _entity("E2", (2, 3), "I11")]
    pred = [
        _entity("P1", (0, 1), None, candidate_code="I10", candidate_score=0.8),
        _entity("P2", (2, 3), None, candidate_code="I12", candidate_score=0.6),
    ]

    curve = coverage_accuracy_curve(gold, pred, thresholds=(0.5, 0.7, 0.9))

    assert curve == [
        {"threshold": 0.5, "eligible": 2, "covered": 2, "coverage": 1.0, "accuracy": 0.5},
        {"threshold": 0.7, "eligible": 2, "covered": 1, "coverage": 0.5, "accuracy": 1.0},
        {"threshold": 0.9, "eligible": 2, "covered": 0, "coverage": 0.0, "accuracy": 0.0},
    ]


def _entity(
    entity_id: str,
    span: tuple[int, int],
    code: str | None,
    *,
    candidate_code: str | None = None,
    candidate_score: float = 0.0,
) -> EntityAnnotation:
    candidates = (
        [
            CandidateConcept(
                code_system=CodeSystem.ICD10,
                code=candidate_code,
                name=candidate_code or "",
                retrieval_score=candidate_score,
                emit_probability=candidate_score,
                concept_id=f"ICD-10:{candidate_code}",
                source="test",
                evidence_sources=("test",),
                matched_alias=entity_id,
                qualified=True,
                qualification_reason="test_candidate",
            )
        ]
        if candidate_code is not None
        else []
    )
    return EntityAnnotation(
        id=entity_id,
        span=span,
        text=entity_id,
        normalized_text=entity_id.lower(),
        type=EntityType.DISEASE,
        code_system=CodeSystem.ICD10 if code else CodeSystem.NONE,
        code=code,
        candidates=candidates,
    )
