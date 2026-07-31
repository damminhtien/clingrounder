"""Tests for turning token model offsets into joint lattice source evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from medical_kg_nlp.benchmarks.phase1.joint_span_sources import (
    build_phase1_medication_parser_source_rows,
    build_phase1_joint_span_proposal_matrix,
    build_phase1_rule_source_rows,
    build_phase1_token_model_proposal_rows,
    load_phase1_joint_span_source_rows,
)
from medical_kg_nlp.benchmarks.phase1.joint_span_token_source import (
    Phase1TokenSourceConfig,
    materialize_phase1_token_model_source,
)
from medical_kg_nlp.benchmarks.phase1.proposal_features import ProposalSourceRole
from medical_kg_nlp.benchmarks.phase1.reviewed_corpus import Phase1ReviewedCorpus
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.utils.text import normalize_for_match
from medical_kg_nlp.utils.hashing import sha256_directory


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


def test_medication_parser_source_emits_only_a_valid_full_sig_span() -> None:
    source = "1. aspirin 81 mg po daily điều trị đau"
    corpus = Phase1ReviewedCorpus(
        source_texts={"1": source},
        gold_rows={"1": ()},
        split_by_document={"1": "train"},
    )
    dictionary = DictionaryStore(
        (
            ConceptEntry(
                concept_id="rx:aspirin",
                code="1191",
                code_system=CodeSystem.RXNORM,
                canonical_name="aspirin",
                semantic_type=EntityType.DRUG,
                aliases=("aspirin",),
            ),
        )
    )

    rows = build_phase1_medication_parser_source_rows(corpus, dictionary)

    assert [(row["text"], row["position"]) for row in rows["1"]] == [
        ("aspirin 81 mg po daily", [3, 25])
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


def test_token_source_artifact_is_complete_and_reloads_with_raw_offsets(tmp_path: Path) -> None:
    corpus = Phase1ReviewedCorpus(
        source_texts={"1": "đau ngực", "authorized_gt:1": "ho"},
        gold_rows={"1": (), "authorized_gt:1": ()},
        split_by_document={"1": "train", "authorized_gt:1": "train"},
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "xlmr-source"

    report = materialize_phase1_token_model_source(
        corpus,
        Phase1TokenSourceConfig(
            model_path=model,
            model_fingerprint=sha256_directory(model),
            model_id="FacebookAI/xlm-roberta-base",
            base_revision="e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
        ),
        output_dir=output,
        extractor=_StaticExtractor(),
    )

    assert report["document_count"] == 2
    assert report["proposal_count"] == 2
    assert load_phase1_joint_span_source_rows(output, corpus)["1"][0]["text"] == "đau ngực"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provenance"]["checkpoint_sha256"] == sha256_directory(model)


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
