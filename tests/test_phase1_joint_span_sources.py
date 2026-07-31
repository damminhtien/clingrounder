"""Tests for turning token model offsets into joint lattice source evidence."""

from __future__ import annotations

from dataclasses import dataclass

from medical_kg_nlp.benchmarks.phase1.joint_span_sources import (
    build_phase1_joint_span_proposal_matrix,
    build_phase1_token_model_proposal_rows,
)
from medical_kg_nlp.benchmarks.phase1.proposal_features import ProposalSourceRole
from medical_kg_nlp.benchmarks.phase1.reviewed_corpus import Phase1ReviewedCorpus
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.utils.text import normalize_for_match


def test_token_model_rows_preserve_raw_offsets_and_matrix_source_evidence() -> None:
    corpus = Phase1ReviewedCorpus(
        source_texts={"1": "đau ngực"},
        gold_rows={"1": ()},
        split_by_document={"1": "train"},
    )
    model_rows = build_phase1_token_model_proposal_rows(
        corpus,
        _StaticExtractor(),
        source_name="xlmr",
    )
    rule_rows = {
        "1": (
            {
                "document_id": "1",
                "proposal_id": "rule-1",
                "text": "đau ngực",
                "type": "TRIỆU_CHỨNG",
                "position": [0, 8],
                "confidence": 0.8,
                "source_label": "SYMPTOM",
            },
        )
    }

    matrix = build_phase1_joint_span_proposal_matrix(
        corpus,
        {"rule": rule_rows, "xlmr": model_rows},
        source_roles={"rule": ProposalSourceRole.RULE, "xlmr": ProposalSourceRole.TOKEN_MODEL},
    )

    assert model_rows["1"][0]["position"] == [0, 8]
    assert matrix["1"][0]["sources"] == ["rule", "xlmr"]
    assert matrix["1"][0]["source_count"] == 2


@dataclass(frozen=True)
class _StaticExtractor:
    def extract(self, source_text: str) -> list[EntityAnnotation]:
        return [
            EntityAnnotation(
                id="M1",
                span=(0, len(source_text)),
                text=source_text,
                normalized_text=normalize_for_match(source_text),
                type=EntityType.SYMPTOM,
                confidence=0.9,
            )
        ]
