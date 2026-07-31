"""Tests for turning token model offsets into joint lattice source evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from medical_kg_nlp.benchmarks.phase1.joint_span_sources import (
    build_phase1_joint_span_proposal_matrix,
    build_phase1_rule_source_rows,
    build_phase1_token_model_proposal_rows,
    load_phase1_joint_span_source_rows,
)
from medical_kg_nlp.benchmarks.phase1.proposal_features import ProposalSourceRole
from medical_kg_nlp.benchmarks.phase1.reviewed_corpus import Phase1ReviewedCorpus
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.types import CodeSystem, EntityType
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


def test_rule_source_rows_preserve_ambiguous_types_for_joint_verification() -> None:
    corpus = Phase1ReviewedCorpus(
        source_texts={"1": "Bệnh nhân chóng mặt"},
        gold_rows={"1": ()},
        split_by_document={"1": "train"},
    )
    dictionary = DictionaryStore(
        (
            ConceptEntry(
                concept_id="D:R42",
                code="R42",
                code_system=CodeSystem.ICD10,
                canonical_name="chóng mặt",
                semantic_type=EntityType.DISEASE,
                aliases=("chóng mặt",),
            ),
            ConceptEntry(
                concept_id="S:R42",
                code="R42",
                code_system=CodeSystem.LOCAL,
                canonical_name="chóng mặt",
                semantic_type=EntityType.SYMPTOM,
                aliases=("chóng mặt",),
            ),
        )
    )

    rows = build_phase1_rule_source_rows(corpus, dictionary)

    assert [(row["text"], row["type"]) for row in rows["1"]] == [
        ("chóng mặt", "CHẨN_ĐOÁN"),
        ("chóng mặt", "TRIỆU_CHỨNG"),
    ]


def test_loader_supports_governed_source_prefixed_document_ids(tmp_path: Path) -> None:
    corpus = Phase1ReviewedCorpus(
        source_texts={"authorized_gt:1": "Bệnh nhân ho"},
        gold_rows={"authorized_gt:1": ()},
        split_by_document={"authorized_gt:1": "train"},
    )
    consensus = tmp_path / "qwen" / "consensus"
    consensus.mkdir(parents=True)
    (consensus / "authorized_gt:1.json").write_text(
        json.dumps(
            [
                {
                    "text": "ho",
                    "type": "TRIỆU_CHỨNG",
                    "position": [10, 12],
                    "confidence": 0.9,
                    "source_label": "qwen.recall",
                }
            ]
        ),
        encoding="utf-8",
    )

    rows = load_phase1_joint_span_source_rows(tmp_path / "qwen", corpus)

    assert rows["authorized_gt:1"][0]["text"] == "ho"


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
