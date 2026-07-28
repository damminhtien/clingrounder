"""Calibration and runtime projection tests for the Phase 1 proposal verifier."""

from __future__ import annotations

from medical_kg_nlp.benchmarks.phase1.proposal_calibration import (
    Phase1ProposalVerifier,
    fit_phase1_proposal_verifier,
    score_phase1_proposal_rows,
)
from medical_kg_nlp.benchmarks.phase1.proposal_dataset import (
    Phase1ProposalDataset,
    Phase1ProposalExample,
)
from medical_kg_nlp.benchmarks.phase1.proposal_features import ProposalSourceRole
from medical_kg_nlp.ontology.phase1 import PHASE1_ALLOWED_TYPES


def test_proposal_verifier_learns_calibrates_and_round_trips() -> None:
    dataset = _dataset()

    verifier, report = fit_phase1_proposal_verifier(dataset)
    restored = Phase1ProposalVerifier.from_dict(verifier.to_dict())

    assert set(restored.threshold_by_type) == set(PHASE1_ALLOWED_TYPES)
    assert report["holdout_opened"] is False
    assert report["development_selection"]["learned"]["f1"] == 1.0
    assert report["development_selection"]["learned"]["false_positive"] == 0
    assert report["coverage_ceiling"]["development"]["recall"] == 1.0


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
            "feature_contract": "phase1-proposal-features.v1",
            "gold_entity_counts": gold_counts,
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
