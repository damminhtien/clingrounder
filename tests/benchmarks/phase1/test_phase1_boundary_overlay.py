"""Safety contracts for max-score conservative boundary replacement."""

from __future__ import annotations

from clingrounder.benchmarks.phase1.boundary_overlay import (
    BoundaryPolicy,
    apply_conservative_boundary_overlay,
)
from clingrounder.benchmarks.phase1.boundary_variants import (
    Phase1BoundaryVariant,
)
from clingrounder.benchmarks.phase1.boundary_verifier import (
    Phase1BoundaryVerifier,
    ScoredPhase1BoundaryVariant,
)
from clingrounder.evaluation.sparse_logistic import SparseLogisticModel
from clingrounder.benchmarks.phase1.ontology import PHASE1_ALLOWED_TYPES


def test_exact_boundary_is_not_replaced() -> None:
    text = "đau ngực"
    base, exact = _scored_pair(
        text,
        base_text="đau ngực",
        variant_text="đau ngực",
        variant_selected=False,
    )

    result = _apply(text, _row("đau ngực", "TRIỆU_CHỨNG", 0), exact, base)

    assert result.rows_by_document["1"][0]["text"] == "đau ngực"
    assert result.changed_identities == frozenset()
    assert result.counters["replacement.changed_identity_count"] == 0


def test_too_short_boundary_can_expand() -> None:
    text = "đau ngực"
    base, expanded = _scored_pair(
        text,
        base_text="đau",
        variant_text="đau ngực",
    )

    result = _apply(text, _row("đau", "TRIỆU_CHỨNG", 0), expanded, base)

    row = result.rows_by_document["1"][0]
    assert row["text"] == "đau ngực"
    assert row["position"] == [0, len(text)]
    assert result.counters["replacement.applied"] == 1


def test_too_long_boundary_can_shrink() -> None:
    text = "đau ngực kéo dài"
    base, shortened = _scored_pair(
        text,
        base_text="đau ngực kéo dài",
        variant_text="đau ngực",
    )

    result = _apply(
        text,
        _row("đau ngực kéo dài", "TRIỆU_CHỨNG", 0),
        shortened,
        base,
    )

    row = result.rows_by_document["1"][0]
    assert row["text"] == "đau ngực"
    assert row["position"] == [0, len("đau ngực")]


def test_low_margin_replacement_is_rejected() -> None:
    text = "đau ngực"
    base, expanded = _scored_pair(
        text,
        base_text="đau",
        variant_text="đau ngực",
        base_probability=0.74,
        variant_probability=0.80,
    )

    result = _apply(text, _row("đau", "TRIỆU_CHỨNG", 0), expanded, base)

    assert result.rows_by_document["1"][0]["text"] == "đau"
    assert result.counters["replacement.rejected.below_replacement_margin"] == 1


def test_changed_identity_does_not_copy_old_candidates() -> None:
    text = "tăng huyết áp nặng"
    base, expanded = _scored_pair(
        text,
        base_text="tăng huyết áp",
        variant_text="tăng huyết áp nặng",
        entity_type="CHẨN_ĐOÁN",
    )
    base_row = _row("tăng huyết áp", "CHẨN_ĐOÁN", 0)
    base_row["candidates"] = ["I10"]
    base_row["assertions"] = ["isHistorical"]

    result = _apply(text, base_row, expanded, base)

    row = result.rows_by_document["1"][0]
    assert row["text"] == "tăng huyết áp nặng"
    assert row["candidates"] == []
    assert row["assertions"] == []


def test_heading_variant_is_rejected() -> None:
    text = "Triệu chứng hiện tại:\nđau ngực"
    base, heading = _scored_pair(
        text,
        base_text="Triệu chứng",
        variant_text="Triệu chứng hiện tại",
    )

    result = _apply(
        text,
        _row("Triệu chứng", "TRIỆU_CHỨNG", 0),
        heading,
        base,
    )

    assert result.rows_by_document["1"][0]["text"] == "Triệu chứng"
    assert result.counters["replacement.rejected.structural_heading"] == 1


def test_offset_remains_exact_raw_substring() -> None:
    text = "Bệnh nhân đau ngực."
    start = text.index("đau")
    base, expanded = _scored_pair(
        text,
        base_text="đau",
        variant_text="đau ngực",
        start=start,
    )

    result = _apply(text, _row("đau", "TRIỆU_CHỨNG", start), expanded, base)

    row = result.rows_by_document["1"][0]
    start, end = row["position"]
    assert text[start:end] == row["text"]


def _apply(
    text: str,
    base_row: dict[str, object],
    replacement: ScoredPhase1BoundaryVariant,
    base: ScoredPhase1BoundaryVariant,
):
    boundary_row = {
        "text": replacement.variant.text,
        "type": replacement.variant.entity_type,
        "assertions": [],
        "position": list(replacement.variant.position),
    }
    if replacement.variant.entity_type == "CHẨN_ĐOÁN":
        boundary_row["candidates"] = []
    return apply_conservative_boundary_overlay(
        base_rows={"1": [base_row]},
        boundary_rows={"1": [boundary_row]},
        scored_variants=(base, replacement),
        source_text_by_document={"1": text},
        verifier=_verifier(),
        policy=BoundaryPolicy(
            mode="conservative_replacement",
            require_same_type=True,
            require_base_selected=True,
            preserve_unmodified_identity=True,
        ),
    )


def _scored_pair(
    source_text: str,
    *,
    base_text: str,
    variant_text: str,
    entity_type: str = "TRIỆU_CHỨNG",
    start: int = 0,
    base_probability: float = 0.60,
    variant_probability: float = 0.95,
    variant_selected: bool = True,
) -> tuple[ScoredPhase1BoundaryVariant, ScoredPhase1BoundaryVariant]:
    base_span = (start, start + len(base_text))
    variant_span = (start, start + len(variant_text))
    base_variant = _variant(
        source_text,
        text=base_text,
        entity_type=entity_type,
        position=base_span,
        foundation=base_span,
        variant_id="base",
    )
    replacement_variant = _variant(
        source_text,
        text=variant_text,
        entity_type=entity_type,
        position=variant_span,
        foundation=base_span,
        variant_id="replacement",
    )
    return (
        ScoredPhase1BoundaryVariant(
            variant=base_variant,
            genre="unknown",
            probability=base_probability,
            threshold=0.20,
            family_winner=False,
            selected_before_overlap=True,
            selected=False,
            rejection_reason="replaced",
            resolution_policy="conservative_replacement",
            base_selected=True,
            replacement_selected=False,
        ),
        ScoredPhase1BoundaryVariant(
            variant=replacement_variant,
            genre="unknown",
            probability=variant_probability,
            threshold=0.20,
            family_winner=True,
            selected_before_overlap=variant_selected,
            selected=variant_selected,
            rejection_reason=None if variant_selected else "base_identity_wins",
            resolution_policy="conservative_replacement",
            base_selected=False,
            replacement_selected=variant_selected,
        ),
    )


def _variant(
    source_text: str,
    *,
    text: str,
    entity_type: str,
    position: tuple[int, int],
    foundation: tuple[int, int],
    variant_id: str,
) -> Phase1BoundaryVariant:
    assert source_text[position[0] : position[1]] == text
    return Phase1BoundaryVariant(
        document_id="1",
        variant_id=variant_id,
        family_id="family",
        text=text,
        entity_type=entity_type,
        position=position,
        sources=("qwen", "rule"),
        source_evidence=(
            ("qwen", 0.9, (), False),
            ("rule", 0.9, (), False),
        ),
        generators=("proposal",),
        foundation_spans=(foundation,),
        status="boundary_variant",
        all_source_agreement=True,
    )


def _row(text: str, entity_type: str, start: int) -> dict[str, object]:
    row: dict[str, object] = {
        "text": text,
        "type": entity_type,
        "assertions": [],
        "position": [start, start + len(text)],
    }
    if entity_type == "CHẨN_ĐOÁN":
        row["candidates"] = []
    return row


def _verifier() -> Phase1BoundaryVerifier:
    return Phase1BoundaryVerifier(
        model=SparseLogisticModel(feature_names=(), weights=(), bias=0.0),
        thresholds=tuple(
            (entity_type, 0.75) for entity_type in sorted(PHASE1_ALLOWED_TYPES)
        ),
        genre_thresholds=(),
        replacement_margins=tuple(
            (entity_type, 0.20) for entity_type in sorted(PHASE1_ALLOWED_TYPES)
        ),
        resolution_policy="conservative_replacement",
        training_dataset_sha256="a" * 64,
        requires_base_probability=True,
    )
