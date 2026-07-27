"""Contracts for Vietnamese model evidence in the Qwen Phase 1 campaign."""

from __future__ import annotations

from pathlib import Path

from medical_kg_nlp.adapters.model_spans import ProjectedSourceEntity
from medical_kg_nlp.benchmarks.phase1.qwen_proposals import (
    select_qwen_confirmed_proposals,
)
from medical_kg_nlp.benchmarks.phase1.qwen_run_spec import (
    load_phase1_qwen_run_spec,
)
from medical_kg_nlp.benchmarks.phase1.vietnamese_support import (
    load_phase1_vietnamese_support_spec,
    project_vietnamese_support_rows,
)
from medical_kg_nlp.ner.proposal import EntityProposal
from medical_kg_nlp.schema.types import EntityType

ROOT = Path(__file__).resolve().parents[1]


def test_vietmed_verifier_and_qwen_remain_below_nine_billion_parameters() -> None:
    support = load_phase1_vietnamese_support_spec(
        ROOT / "configs/models/phase1-vietmed-ner-verifier-2026-07-27.yaml"
    )
    qwen = load_phase1_qwen_run_spec(
        ROOT
        / "configs/models/phase1-qwen3-8b-vietmed-verifier-2026-07-27.yaml"
    )

    assert support.model.parameter_count == 277_481_509
    assert qwen.budget.total_parameters == 8_468_216_869
    assert qwen.budget.total_parameters < 9_000_000_000
    assert support.model in qwen.budget.entries


def test_vietmed_broad_label_projects_to_support_candidates_only() -> None:
    text = "bệnh nhân đau đầu và dùng aspirin"
    rows = project_vietnamese_support_rows(
        text,
        (
            ProjectedSourceEntity(
                span=(10, 17),
                source_label="DISEASESYMTOM",
                confidence=0.91,
            ),
            ProjectedSourceEntity(
                span=(26, 33),
                source_label="DRUGCHEMICAL",
                confidence=0.95,
            ),
        ),
        compatibility_map={
            "DISEASESYMTOM": ("CHẨN_ĐOÁN", "TRIỆU_CHỨNG"),
            "DRUGCHEMICAL": ("THUỐC",),
        },
    )

    assert [(row["text"], row["type"]) for row in rows] == [
        ("đau đầu", "CHẨN_ĐOÁN"),
        ("đau đầu", "TRIỆU_CHỨNG"),
        ("aspirin", "THUỐC"),
    ]
    assert all(row["support_only"] is True for row in rows)
    assert all(row["assertions"] == [] and row["candidates"] == [] for row in rows)


def test_vietmed_support_cannot_enter_consensus_without_qwen() -> None:
    support = EntityProposal(
        span=(0, 7),
        candidate_types=(EntityType.SYMPTOM,),
        source="vietmed.ner",
        score=0.99,
    )
    rule = EntityProposal(
        span=(0, 7),
        candidate_types=(EntityType.SYMPTOM,),
        source="rule",
        score=0.99,
    )

    selected = select_qwen_confirmed_proposals(
        {"vietmed.ner": (support,), "rule": (rule,)},
        thresholds={EntityType.SYMPTOM: 0.7},
    )

    assert selected == ()
