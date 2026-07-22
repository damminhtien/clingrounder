"""Round 2 audit must remain provenance-only and runtime-ineligible."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from medical_kg_nlp.benchmarks.phase1.round2 import (
    build_phase1_round2_audit,
    load_phase1_round2_documents,
    write_phase1_round2_audit,
)
from medical_kg_nlp.mining.io import write_jsonl
from medical_kg_nlp.mining.records import (
    AccessClass,
    MinedDocument,
    RedistributionPolicy,
)


def _document(source_id: str, text: str) -> MinedDocument:
    return MinedDocument(
        document_id=f"round2:{source_id}",
        text=text,
        language="vi",
        note_type="mixed_medical_text",
        source_artifact_id="round2:archive",
        access_class=AccessClass.LOCAL_PRIVATE,
        redistribution=RedistributionPolicy.PROHIBITED,
        hosted_processing_allowed=False,
        metadata={
            "archive_member": f"input/{source_id}.txt",
            "source_document_id": source_id,
            "source_archive_sha256": "a" * 64,
            "raw_bytes_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "parser_id": "plain_text_archive",
            "newline_normalization": "none",
        },
    )


def test_round2_manifest_loader_preserves_private_raw_text_contract() -> None:
    mined = tuple(_document(str(index), f"Dòng {index}\r\nDòng hai") for index in range(1, 101))

    documents = load_phase1_round2_documents(
        mined,
        expected_archive_sha256="a" * 64,
    )

    assert [document.document_id for document in documents] == [
        str(index) for index in range(1, 101)
    ]
    assert documents[0].text == "Dòng 1\r\nDòng hai"
    assert documents[0].metadata["archive_member"] == "input/1.txt"
    assert documents[0].metadata["access_class"] == "local_private"


def test_round2_audit_emits_no_runtime_annotation_memory(tmp_path: Path) -> None:
    reference_input = tmp_path / "reference"
    reference_gold = tmp_path / "gold"
    reference_input.mkdir()
    reference_gold.mkdir()
    (reference_input / "1.txt").write_text("Tiền sử bệnh\nSốt cao kéo dài", encoding="utf-8")
    (reference_input / "2.txt").write_text("Ho và khó thở", encoding="utf-8")
    (reference_gold / "1.json").write_text(
        json.dumps(
            [
                {
                    "text": "Sốt cao",
                    "position": [14, 21],
                    "type": "TRIỆU_CHỨNG",
                    "assertions": [],
                    "candidates": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    (reference_gold / "2.json").write_text("[]", encoding="utf-8")
    split_manifest = tmp_path / "split.json"
    split_manifest.write_text(
        json.dumps(
            {
                "assignments": [
                    {"document_id": "1", "split": "train"},
                    {"document_id": "2", "split": "holdout"},
                ],
                "corpus": {"fingerprint_sha256": "b" * 64},
            }
        ),
        encoding="utf-8",
    )
    documents = (
        _document("1", "Tiền sử bệnh\nSốt cao kéo dài và mệt"),
        _document("2", "Câu hỏi từ người dùng: đau đầu"),
    )

    audit = build_phase1_round2_audit(
        documents,
        reference_input_dir=reference_input,
        reference_gold_dir=reference_gold,
        reference_split_manifest=split_manifest,
        novelty_source_ids=("2",),
    )

    assert audit["policy"] == {
        "purpose": "distribution_and_novelty_audit_only",
        "runtime_eligible": False,
        "contains_source_text": False,
        "contains_annotations": False,
        "contains_candidates": False,
    }
    duplicate = audit["duplicate_report"]
    assert duplicate["policy"]["annotation_transfer_permitted"] is False
    assert duplicate["cross_round"]["exact_line_document_count"] == 1
    assert audit["novelty_queue"][0]["source_document_id"] == "2"
    assert audit["novelty_queue"][0]["runtime_eligible"] is False
    annotation_profile = audit["profile"]["annotations"]
    assert annotation_profile["count"] == 0
    assert annotation_profile["entity_types"] == {}
    assert annotation_profile["source_labels"] == {}
    serialized = json.dumps(audit, ensure_ascii=False)
    assert "Sốt cao" not in serialized
    assert '"candidates":' not in serialized
    assert '"position":' not in serialized


def test_round2_audit_writer_is_deterministic(tmp_path: Path) -> None:
    documents_path = tmp_path / "documents.jsonl"
    write_jsonl(documents_path, (_document("1", "Sốt").to_dict(),))
    audit = {
        "profile": {
            "round2": {"source_archive_sha256": "a" * 64},
        },
        "duplicate_report": {"runtime_eligible": False},
        "novelty_queue": (
            {"source_document_id": "1", "runtime_eligible": False},
        ),
        "reference": {"corpus_fingerprint_sha256": "b" * 64},
    }

    first = write_phase1_round2_audit(
        audit,
        tmp_path / "audit",
        documents_manifest_path=documents_path,
    )
    second = write_phase1_round2_audit(
        audit,
        tmp_path / "audit",
        documents_manifest_path=documents_path,
    )

    assert first == second
    assert first["runtime_eligible"] is False
    assert set(first["outputs"]) == {
        "profile.json",
        "duplicate_report.json",
        "novelty_queue.jsonl",
    }
