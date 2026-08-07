"""Joint span training data is proposal-derived, raw-offset safe, and source-governed."""

from __future__ import annotations

from clingrounder.benchmarks.phase1.joint_span_dataset import (
    build_phase1_joint_span_dataset,
)
from clingrounder.benchmarks.phase1.proposal_features import ProposalSourceRole
from clingrounder.benchmarks.phase1.reviewed_corpus import Phase1ReviewedCorpus


def test_joint_span_dataset_reports_missing_gold_without_gold_seeding() -> None:
    corpus = Phase1ReviewedCorpus(
        source_texts={"1": "đau ngực; buồn nôn"},
        gold_rows={
            "1": (
                {"text": "đau ngực", "type": "TRIỆU_CHỨNG", "position": [0, 8]},
                {"text": "buồn nôn", "type": "TRIỆU_CHỨNG", "position": [10, 18]},
            )
        },
        split_by_document={"1": "train"},
    )
    dataset = build_phase1_joint_span_dataset(
        corpus,
        {"1": [_row()]},
        source_roles={"rule:dictionary": ProposalSourceRole.RULE},
        source_dataset_by_document={"1": "manual_gold"},
    )

    assert dataset.manifest["candidate_coverage"] == {
        "covered_gold": 1,
        "gold": 2,
        "recall": 0.5,
    }
    assert any(example.label.value == "EXACT_SYMPTOM" for example in dataset.examples)
    assert all(example.candidate.variant.text != "buồn nôn" for example in dataset.examples)


def test_joint_span_dataset_coverage_counts_unique_gold_identity_once() -> None:
    corpus = Phase1ReviewedCorpus(
        source_texts={"1": "đau ngực"},
        gold_rows={
            "1": (
                {"text": "đau ngực", "type": "TRIỆU_CHỨNG", "position": [0, 8]},
                {"text": "đau ngực", "type": "TRIỆU_CHỨNG", "position": [0, 8]},
            )
        },
        split_by_document={"1": "train"},
    )

    dataset = build_phase1_joint_span_dataset(
        corpus,
        {"1": [{**_row(), "text": "đau ngực", "position": [0, 8]}]},
        source_roles={"rule:dictionary": ProposalSourceRole.RULE},
        source_dataset_by_document={"1": "manual_gold"},
    )

    assert dataset.manifest["candidate_coverage"] == {
        "covered_gold": 1,
        "gold": 1,
        "recall": 1.0,
    }


def _row() -> dict[str, object]:
    return {
        "document_id": "1",
        "proposal_id": "1:rule:0",
        "text": "đau",
        "type": "TRIỆU_CHỨNG",
        "position": [0, 3],
        "sources": ["rule:dictionary"],
        "source_count": 1,
        "all_source_agreement": False,
        "status": "source_only",
        "source_evidence": {
            "rule:dictionary": {
                "confidence": 0.8,
                "source_labels": ["SYMPTOM"],
                "support_only": False,
            }
        },
    }
