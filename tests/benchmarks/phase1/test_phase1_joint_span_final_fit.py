"""Contracts for governed final-fit joint span/type dataset preparation."""

from __future__ import annotations

import json
from pathlib import Path

from clingrounder.benchmarks.phase1.final_supervision import (
    Phase1FinalSupervisionCorpus,
)
from clingrounder.benchmarks.phase1.joint_span_final_fit import (
    prepare_phase1_joint_span_final_fit,
)
from clingrounder.benchmarks.phase1.proposal_features import ProposalSourceRole
from clingrounder.benchmarks.phase1.reviewed_corpus import Phase1ReviewedCorpus
from clingrounder.dictionaries.dictionary_store import DictionaryStore
from clingrounder.dictionaries.synonym_table import ConceptEntry
from clingrounder.schema.types import CodeSystem, EntityType


def test_final_fit_preparation_aligns_rule_and_qwen_with_provenance(tmp_path: Path) -> None:
    corpus = Phase1FinalSupervisionCorpus(
        reviewed=Phase1ReviewedCorpus(
            source_texts={"1": "Bệnh nhân ho", "authorized_gt:1": "Bệnh nhân sốt"},
            gold_rows={
                "1": (_row("ho", 10, 12),),
                "authorized_gt:1": (_row("sốt", 10, 13),),
            },
            split_by_document={"1": "train", "authorized_gt:1": "train"},
        ),
        source_by_document={"1": "manual_gold", "authorized_gt:1": "authorized_ground_truth"},
        manifest={"fingerprint_sha256": "f" * 64},
    )
    dictionary = DictionaryStore(
        (
            ConceptEntry(
                concept_id="symptom:ho",
                code="SYM-HO",
                code_system=CodeSystem.LOCAL,
                canonical_name="ho",
                semantic_type=EntityType.SYMPTOM,
                aliases=("ho",),
            ),
            ConceptEntry(
                concept_id="symptom:sot",
                code="SYM-SOT",
                code_system=CodeSystem.LOCAL,
                canonical_name="sốt",
                semantic_type=EntityType.SYMPTOM,
                aliases=("sốt",),
            ),
        )
    )
    qwen = tmp_path / "qwen"
    consensus = qwen / "consensus"
    consensus.mkdir(parents=True)
    _write_source(consensus / "1.json", "ho", 10, 12)
    _write_source(consensus / "authorized_gt:1.json", "sốt", 10, 13)
    (qwen / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "phase1-joint-span-source.v1",
                "document_count": 2,
                "round2_included": False,
                "friend31_included": False,
            }
        ),
        encoding="utf-8",
    )

    report = prepare_phase1_joint_span_final_fit(
        corpus,
        dictionary,
        model_sources={"qwen": (qwen, ProposalSourceRole.LLM)},
        output_dir=tmp_path / "prepared",
    )

    assert report["sources"]["qwen"]["role"] == "llm"
    assert report["sources"]["dictionary_trie"]["kind"] == "independent_typed_lexical_proposals"
    assert report["policy"]["round2_included"] is False
    assert report["policy"]["friend31_included"] is False
    assert Path(report["dataset"]["examples_path"]).is_file()
    assert json.loads((tmp_path / "prepared" / "manifest.json").read_text(encoding="utf-8"))["candidate_coverage"]["covered_gold"] == 2


def test_final_fit_preparation_rejects_verifier_only_source(tmp_path: Path) -> None:
    corpus = Phase1FinalSupervisionCorpus(
        reviewed=Phase1ReviewedCorpus(
            source_texts={"1": "ho"},
            gold_rows={"1": ()},
            split_by_document={"1": "train"},
        ),
        source_by_document={"1": "manual_gold"},
        manifest={},
    )

    try:
        prepare_phase1_joint_span_final_fit(
            corpus,
            DictionaryStore(()),
            model_sources={"vietmed": (tmp_path, ProposalSourceRole.VERIFIER)},
            output_dir=tmp_path / "prepared",
        )
    except ValueError as error:
        assert "Verifier-only" in str(error)
    else:  # pragma: no cover - documents the required rejection.
        raise AssertionError("Verifier-only source must be rejected")


def test_final_fit_preparation_rejects_round2_source_provenance(tmp_path: Path) -> None:
    corpus = Phase1FinalSupervisionCorpus(
        reviewed=Phase1ReviewedCorpus(
            source_texts={"1": "ho"},
            gold_rows={"1": (_row("ho", 0, 2),)},
            split_by_document={"1": "train"},
        ),
        source_by_document={"1": "manual_gold"},
        manifest={},
    )
    source = tmp_path / "invalid"
    consensus = source / "consensus"
    consensus.mkdir(parents=True)
    _write_source(consensus / "1.json", "ho", 0, 2)
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "phase1-joint-span-source.v1",
                "document_count": 1,
                "round2_included": True,
                "friend31_included": False,
            }
        ),
        encoding="utf-8",
    )

    try:
        prepare_phase1_joint_span_final_fit(
            corpus,
            DictionaryStore(()),
            model_sources={"qwen": (source, ProposalSourceRole.LLM)},
            output_dir=tmp_path / "prepared",
        )
    except ValueError as error:
        assert "round2_included=true" in str(error)
    else:  # pragma: no cover - documents the required rejection.
        raise AssertionError("Round 2 source provenance must be rejected")


def _row(text: str, start: int, end: int) -> dict[str, object]:
    return {
        "text": text,
        "type": "TRIỆU_CHỨNG",
        "position": [start, end],
        "assertions": [],
        "candidates": [],
    }


def _write_source(path: Path, text: str, start: int, end: int) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "text": text,
                    "type": "TRIỆU_CHỨNG",
                    "position": [start, end],
                    "confidence": 0.9,
                    "source_label": "qwen.recall",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
