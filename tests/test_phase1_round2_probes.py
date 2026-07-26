"""Focused contracts for isolated Round 2 assertion and entity probes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from medical_kg_nlp.benchmarks.phase1.phase1 import zip_phase1_output_dir
from medical_kg_nlp.benchmarks.phase1.phase1_ensemble import (
    load_phase1_output_source,
)
from medical_kg_nlp.benchmarks.phase1.round2_probes import (
    Phase1Round2ProbeConfig,
    align_quoted_phase1_proposals,
    merge_region_routed_proposals,
    run_phase1_round2_probes,
    segment_phase1_text_regions,
)
from medical_kg_nlp.cli.parser import build_parser
from medical_kg_nlp.mining.io import write_jsonl
from medical_kg_nlp.mining.records import (
    AccessClass,
    MinedDocument,
    RedistributionPolicy,
)
from medical_kg_nlp.utils.hashing import sha256_file


def test_round2_region_segmentation_preserves_raw_offsets() -> None:
    text = (
        "Câu hỏi từ người dùng:\n"
        "Tôi bị đau đầu.\n"
        "Câu trả lời của bác sĩ:\n"
        "Đau đầu có nhiều nguyên nhân.\n"
        "Tiền sử bệnh:\n"
        "Tăng huyết áp.\n"
        "Danh sách thuốc trước nhập viện:\n"
        "1. aspirin 81 mg po daily"
    )

    regions = segment_phase1_text_regions(text)

    assert [region.kind for region in regions] == [
        "question_answer",
        "clinical",
        "medication_list",
    ]
    assert "".join(region.text for region in regions) == text
    for region in regions:
        start, end = region.span
        assert text[start:end] == region.text


def test_quoted_proposals_expand_all_exact_occurrences_and_use_context() -> None:
    text = "ho nhẹ; sau đó ho tăng."

    rows, rejected = align_quoted_phase1_proposals(
        text,
        (
            {"text": "ho", "type": "TRIỆU_CHỨNG"},
            {
                "text": "ho",
                "type": "TRIỆU_CHỨNG",
                "left_context": "sau đó ",
            },
            {"text": "khó thở", "type": "TRIỆU_CHỨNG"},
        ),
    )

    assert [(row["text"], row["position"]) for row in rows] == [
        ("ho", [0, 2]),
        ("ho", [15, 17]),
    ]
    assert rejected == [
        {
            "proposal_index": 2,
            "reason": "quote_or_context_not_found",
            "text": "khó thở",
            "type": "TRIỆU_CHỨNG",
        }
    ]
    assert all(text[row["position"][0] : row["position"][1]] == row["text"] for row in rows)


def test_region_router_allows_qa_recall_but_requires_consensus_in_clinical_text() -> None:
    text = (
        "Câu hỏi từ người dùng:\n"
        "Tôi đau đầu và ho kéo dài.\n"
        "Tiền sử bệnh:\n"
        "Tăng huyết áp."
    )
    baseline = {"1": [_row(text, "ho", "TRIỆU_CHỨNG")]}
    qwen = {
        "1": [
            _row(text, "đau đầu", "TRIỆU_CHỨNG"),
            _row(text, "ho kéo dài", "TRIỆU_CHỨNG"),
            _row(text, "Tăng huyết áp", "CHẨN_ĐOÁN"),
        ]
    }
    xlmr = {"1": [_row(text, "Tăng huyết áp", "CHẨN_ĐOÁN")]}

    merged, decisions, counters = merge_region_routed_proposals(
        baseline,
        {"qwen": qwen, "xlmr": xlmr},
        {"1": text},
    )

    assert [(row["text"], row["type"]) for row in merged["1"]] == [
        ("đau đầu", "TRIỆU_CHỨNG"),
        ("ho", "TRIỆU_CHỨNG"),
        ("Tăng huyết áp", "CHẨN_ĐOÁN"),
    ]
    added = [row for row in merged["1"] if row["text"] != "ho"]
    assert all(row["assertions"] == [] and row["candidates"] == [] for row in added)
    assert {decision["reason"] for decision in decisions} == {
        "single_source_region_route",
        "exact_source_consensus",
    }
    assert counters["proposal.blocked_overlap"] == 1


def test_round2_probe_runner_preserves_entities_and_candidates_for_assertions(
    tmp_path: Path,
) -> None:
    archive_sha256 = "a" * 64
    source_rows = (
        _document("1", "Tiền sử bệnh:\nTăng huyết áp", archive_sha256),
        _document("2", "Triệu chứng hiện tại:\nho", archive_sha256),
    )
    documents_path = tmp_path / "documents.jsonl"
    write_jsonl(documents_path, (document.to_dict() for document in source_rows))

    base_dir = tmp_path / "base"
    base_dir.mkdir()
    base_rows = {
        "1": [_row(source_rows[0].text, "Tăng huyết áp", "CHẨN_ĐOÁN")],
        "2": [_row(source_rows[1].text, "ho", "TRIỆU_CHỨNG")],
    }
    for document_id, rows in base_rows.items():
        (base_dir / f"{document_id}.json").write_text(
            _json(rows),
            encoding="utf-8",
        )
    base_zip = tmp_path / "base.zip"
    zip_phase1_output_dir(base_dir, base_zip)
    dictionary_path = tmp_path / "dictionary.jsonl"
    dictionary_path.write_text("", encoding="utf-8")

    report = run_phase1_round2_probes(
        Phase1Round2ProbeConfig(
            documents_path=documents_path,
            expected_source_archive_sha256=archive_sha256,
            base=base_zip,
            expected_base_sha256=sha256_file(base_zip),
            dictionary_paths=(dictionary_path,),
            output_root=tmp_path / "runs",
            expected_count=2,
        )
    )

    assert [variant["name"] for variant in report["variants"]] == ["A_NEG_HIST"]
    variant = report["variants"][0]
    assert variant["changed"]["assertion_changed"] == 1
    assert variant["changed"]["changed_row_count"] == 1
    assert variant["changed"]["entity_added"] == 0
    assert variant["changed"]["entity_removed"] == 0
    assert variant["changed"].get("candidate_changed", 0) == 0
    output = load_phase1_output_source(variant["zip"])
    assert output["1"][0]["assertions"] == ["isHistorical"]
    assert output["2"][0]["assertions"] == []
    assert [row["candidates"] for rows in output.values() for row in rows] == [[], []]
    manifest = json.loads(Path(report["run_manifest"]).read_text(encoding="utf-8"))
    assert (
        variant["entity_projection_sha256"]
        == manifest["probe_suite"]["baseline"]["entity_projection_sha256"]
    )


def test_round2_probe_cli_parser_accepts_named_sources() -> None:
    args = build_parser().parse_args(
        [
            "benchmark",
            "phase1",
            "round2",
            "probes",
            "--documents",
            "documents.jsonl",
            "--source-archive-sha256",
            "a" * 64,
            "--base",
            "base.zip",
            "--expected-base-sha256",
            "b" * 64,
            "--source",
            "qwen=qwen.zip",
            "--source",
            "xlmr=xlmr.zip",
        ]
    )

    assert args.handler == "benchmark_phase1_round2_probes"
    assert args.source == ["qwen=qwen.zip", "xlmr=xlmr.zip"]


def _row(text: str, mention: str, entity_type: str) -> dict[str, object]:
    start = text.index(mention)
    return {
        "text": mention,
        "type": entity_type,
        "assertions": [],
        "candidates": [],
        "position": [start, start + len(mention)],
    }


def _document(source_id: str, text: str, archive_sha256: str) -> MinedDocument:
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
            "source_archive_sha256": archive_sha256,
            "raw_bytes_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "parser_id": "plain_text_archive",
            "newline_normalization": "none",
        },
    )


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"
