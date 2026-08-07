"""Calibration and runtime projection tests for the Phase 1 proposal verifier."""

from __future__ import annotations

from clingrounder.benchmarks.phase1.proposal_calibration import (
    Phase1ProposalFitMode,
    Phase1ProposalVerifier,
    fit_phase1_proposal_verifier,
    resolve_phase1_proposal_rows,
    score_phase1_proposal_rows,
)
from clingrounder.benchmarks.phase1.proposal_dataset import (
    Phase1ProposalDataset,
    Phase1ProposalExample,
)
from clingrounder.benchmarks.phase1.proposal_features import (
    PHASE1_PROPOSAL_FEATURE_CONTRACT,
    ProposalSourceRole,
)
from clingrounder.benchmarks.phase1.ontology import PHASE1_ALLOWED_TYPES


def test_proposal_verifier_learns_calibrates_and_round_trips() -> None:
    dataset = _dataset()

    verifier, report = fit_phase1_proposal_verifier(dataset)
    restored = Phase1ProposalVerifier.from_dict(verifier.to_dict())

    assert set(restored.threshold_by_type) == set(PHASE1_ALLOWED_TYPES)
    assert report["holdout_opened"] is False
    assert report["development_selection"]["learned"]["f1"] == 1.0
    assert report["development_selection"]["learned"]["false_positive"] == 0
    assert report["coverage_ceiling"]["development"]["recall"] == 1.0
    assert report["probability_calibration"]["selected_method"] in {
        "identity_logistic",
        "platt_document_grouped_oof",
    }
    assert restored.probability_calibrator.fold_count >= 2


def test_runtime_scoring_resolves_overlap_by_probability() -> None:
    verifier, _ = fit_phase1_proposal_verifier(_dataset())
    text = "đau ngực"
    rows = [
        _row("đau ngực", "TRIỆU_CHỨNG", 0, "qwen", "full"),
        _row("ngực", "TRIỆU_CHỨNG", 4, "pipeline", "nested"),
    ]

    scored = score_phase1_proposal_rows(
        rows,
        {"1": text},
        verifier,
        source_roles={
            "pipeline": ProposalSourceRole.RULE,
            "qwen": ProposalSourceRole.LLM,
        },
    )

    selected = [item for item in scored if item.selected]
    assert len(selected) <= 1
    assert all(item.row["proposal_id"] in {"full", "nested"} for item in selected)
    assert all(0.0 <= item.probability <= 1.0 for item in scored)


def test_probability_resolver_emits_only_selected_raw_spans() -> None:
    verifier, _ = fit_phase1_proposal_verifier(_dataset())
    text = "đau ngực"
    rows = [
        _row("đau ngực", "TRIỆU_CHỨNG", 0, "qwen", "full"),
        _row("ngực", "TRIỆU_CHỨNG", 4, "pipeline", "nested"),
    ]

    resolved, scored = resolve_phase1_proposal_rows(
        rows,
        {"1": text},
        verifier,
        source_roles={
            "pipeline": ProposalSourceRole.RULE,
            "qwen": ProposalSourceRole.LLM,
        },
    )

    assert len(resolved["1"]) == sum(item.selected for item in scored)
    assert len(resolved["1"]) <= 1
    for row in resolved["1"]:
        start, end = row["position"]
        assert text[start:end] == row["text"]
        assert row["assertions"] == []


def test_runtime_scoring_blocks_structural_labels_before_overlap() -> None:
    verifier, _ = fit_phase1_proposal_verifier(_dataset())
    text = "Cận lâm sàng:\n"
    rows = [
        _row(
            "Cận lâm sàng",
            "TÊN_XÉT_NGHIỆM",
            0,
            "qwen",
            "heading",
        )
    ]

    scored = score_phase1_proposal_rows(
        rows,
        {"1": text},
        verifier,
        source_roles={
            "pipeline": ProposalSourceRole.RULE,
            "qwen": ProposalSourceRole.LLM,
        },
    )

    assert scored[0].selected_before_overlap is False
    assert scored[0].selected is False
    assert scored[0].rejection_reason == "structural_heading"


def test_precision_operating_point_is_persisted() -> None:
    verifier, report = fit_phase1_proposal_verifier(
        _dataset(),
        minimum_development_precision=0.9,
    )
    restored = Phase1ProposalVerifier.from_dict(verifier.to_dict())

    assert restored.minimum_development_precision == 0.9
    assert report["operating_point"] == {
        "objective": "maximum_recall_at_minimum_precision",
        "minimum_development_precision": 0.9,
    }


def test_full_oof_fit_uses_all_examples_but_cannot_auto_promote() -> None:
    verifier, report = fit_phase1_proposal_verifier(
        _dataset(),
        fit_mode=Phase1ProposalFitMode.FULL_OOF,
    )

    assert report["fit_mode"] == "full_oof"
    assert report["decision_authority"] == "official_submission"
    assert report["local_metrics_role"] == "diagnostic_only"
    assert report["auto_promote"] is False
    assert report["operating_point"]["prediction_source"] == (
        "document_grouped_out_of_fold"
    )
    assert report["training"]["cross_fit"]["document_count"] == 6
    assert set(verifier.threshold_by_type) == set(PHASE1_ALLOWED_TYPES)


def _dataset() -> Phase1ProposalDataset:
    examples = (
        _example("1", "train", "train-positive-a", 1, {"agreement": 1.0}, 0, 8),
        _example("2", "train", "train-positive-b", 1, {"agreement": 1.0}, 0, 4),
        _example("3", "train", "train-negative-a", 0, {"conflict": 1.0}, 0, 4),
        _example("4", "train", "train-negative-b", 0, {"conflict": 1.0}, 0, 3),
        _example("5", "development", "dev-positive", 1, {"agreement": 1.0}, 0, 8),
        _example("6", "development", "dev-negative", 0, {"conflict": 1.0}, 0, 4),
    )
    gold_counts = {
        "train:CHẨN_ĐOÁN": 0,
        "train:KẾT_QUẢ_XÉT_NGHIỆM": 0,
        "train:THUỐC": 0,
        "train:TRIỆU_CHỨNG": 2,
        "train:TÊN_XÉT_NGHIỆM": 0,
        "development:CHẨN_ĐOÁN": 0,
        "development:KẾT_QUẢ_XÉT_NGHIỆM": 0,
        "development:THUỐC": 0,
        "development:TRIỆU_CHỨNG": 1,
        "development:TÊN_XÉT_NGHIỆM": 0,
    }
    return Phase1ProposalDataset(
        examples=examples,
        manifest={
            "feature_contract": PHASE1_PROPOSAL_FEATURE_CONTRACT,
            "gold_entity_counts": gold_counts,
            "gold_entity_genre_counts": {
                "train:unknown:CHẨN_ĐOÁN": 0,
                "train:unknown:KẾT_QUẢ_XÉT_NGHIỆM": 0,
                "train:unknown:THUỐC": 0,
                "train:unknown:TRIỆU_CHỨNG": 2,
                "train:unknown:TÊN_XÉT_NGHIỆM": 0,
                "development:unknown:CHẨN_ĐOÁN": 0,
                "development:unknown:KẾT_QUẢ_XÉT_NGHIỆM": 0,
                "development:unknown:THUỐC": 0,
                "development:unknown:TRIỆU_CHỨNG": 1,
                "development:unknown:TÊN_XÉT_NGHIỆM": 0,
            },
            "source_roles": {
                "pipeline": "rule",
                "qwen": "llm",
            },
        },
    )


def _example(
    document_id: str,
    split: str,
    proposal_id: str,
    label: int,
    features: dict[str, float],
    start: int,
    end: int,
) -> Phase1ProposalExample:
    return Phase1ProposalExample(
        document_id=document_id,
        proposal_id=proposal_id,
        split=split,
        text="x" * (end - start),
        entity_type="TRIỆU_CHỨNG",
        position=(start, end),
        sources=("qwen",) if label else ("pipeline",),
        status="source_only",
        label=label,
        error_kind="exact" if label else "spurious",
        features=tuple(sorted(features.items())),
    )


def _row(
    text: str,
    entity_type: str,
    start: int,
    source: str,
    proposal_id: str,
) -> dict[str, object]:
    return {
        "document_id": "1",
        "proposal_id": proposal_id,
        "text": text,
        "type": entity_type,
        "position": [start, start + len(text)],
        "sources": [source],
        "source_count": 1,
        "all_source_agreement": False,
        "status": "source_only",
    }
