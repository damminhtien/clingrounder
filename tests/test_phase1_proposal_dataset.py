"""Leakage and feature-contract tests for Phase 1 proposal calibration data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from medical_kg_nlp.benchmarks.phase1.proposal_dataset import (
    build_phase1_proposal_dataset,
    write_phase1_proposal_dataset,
)
from medical_kg_nlp.benchmarks.phase1.proposal_features import (
    ProposalSourceRole,
    extract_phase1_proposal_context,
    extract_phase1_proposal_features,
    is_phase1_heading_only_proposal,
)


def test_proposal_features_use_roles_structure_and_bounded_hashes() -> None:
    text = "Thuốc:\n1. aspirin 81 mg po daily\n"
    start = text.index("aspirin")
    row = _proposal(
        "aspirin 81 mg po daily",
        "THUỐC",
        start,
        sources=["pipeline", "qwen"],
        status="exact_agreement",
    )
    row["source_evidence"] = {
        "pipeline": {
            "confidence": 0.82,
            "source_labels": [],
            "support_only": False,
        },
        "qwen": {
            "confidence": 0.91,
            "source_labels": ["SYMPTOM"],
            "support_only": False,
        },
    }
    row["overlap_agreements"] = [
        {
            "position": [start, start + len("aspirin")],
            "text": "aspirin",
            "sources": ["pipeline"],
        }
    ]
    row["type_conflicts"] = [
        {
            "type": "KẾT_QUẢ_XÉT_NGHIỆM",
            "sources": ["qwen"],
        }
    ]

    features = extract_phase1_proposal_features(
        row,
        text,
        {
            "pipeline": ProposalSourceRole.RULE,
            "qwen": ProposalSourceRole.LLM,
        },
    )

    assert features["role:rule"] == 1.0
    assert features["role:llm"] == 1.0
    assert features["numeric:role_confidence_max:rule"] == 0.82
    assert features["numeric:role_confidence_max:llm"] == 0.91
    assert features["numeric:role_source_count:rule"] == 1.0
    assert features["numeric:role_source_count:llm"] == 1.0
    assert features["numeric:overlap_count"] == 1.0
    assert features["numeric:type_conflict_count"] == 1.0
    assert features["numeric:contains_competitor_count"] == 1.0
    assert features["conflict_type:KẾT_QUẢ_XÉT_NGHIỆM"] == 1.0
    assert features["section:medication"] == 1.0
    assert features["flag:list_item"] == 1.0
    assert features["flag:starts_list_item"] == 1.0
    assert features["flag:contains_digit"] == 1.0
    assert features["flag:contains_unit"] == 1.0
    assert any(name.startswith("hash:mention_char_3:") for name in features)
    assert any(name.startswith("hash:source_label:llm:") for name in features)
    assert not any("document_id" in name or "absolute" in name for name in features)
    context = extract_phase1_proposal_context(row, text)
    assert context.section == "medication"
    assert context.genre == "unknown"


def test_heading_only_feature_does_not_block_content_after_colon() -> None:
    text = "Cận lâm sàng:\n- CRP: tăng"
    heading = _proposal(
        "Cận lâm sàng",
        "TÊN_XÉT_NGHIỆM",
        0,
        sources=["qwen"],
        status="source_only",
    )
    result_start = text.index("tăng")
    result = _proposal(
        "tăng",
        "KẾT_QUẢ_XÉT_NGHIỆM",
        result_start,
        sources=["qwen"],
        status="source_only",
    )

    assert is_phase1_heading_only_proposal(heading, text) is True
    assert is_phase1_heading_only_proposal(result, text) is False


def test_heading_only_feature_blocks_labels_inside_heading_line() -> None:
    text = (
        "Điện tâm đồ: ghi tại giường.II. Kết quả xét nghiệm & "
        "Cận lâm sàng đã có\n"
        "Kết quả Cận lâm sàng\n"
    )
    first_start = text.index("Cận lâm sàng")
    second_start = text.index("Cận lâm sàng", first_start + 1)

    for start in (first_start, second_start):
        row = _proposal(
            "Cận lâm sàng",
            "TÊN_XÉT_NGHIỆM",
            start,
            sources=["qwen"],
            status="source_only",
        )
        assert is_phase1_heading_only_proposal(row, text) is True


def test_proposal_dataset_labels_errors_without_reading_holdout(tmp_path: Path) -> None:
    paths = _write_dataset_fixture(tmp_path)

    dataset = build_phase1_proposal_dataset(
        paths["matrix"],
        paths["input"],
        paths["gold"],
        paths["model_manifest"],
        paths["holdout_manifest"],
        source_roles={
            "pipeline": ProposalSourceRole.RULE,
            "qwen": ProposalSourceRole.LLM,
        },
    )

    assert {example.document_id for example in dataset.examples} == {"1", "2"}
    error_by_proposal = {
        example.proposal_id: example.error_kind
        for example in dataset.examples
        if example.document_id == "1"
    }
    assert error_by_proposal == {
        "exact": "exact",
        "type": "type_confusion",
        "boundary": "boundary",
        "boundary-type": "boundary_type_confusion",
        "spurious": "spurious",
    }
    assert dataset.manifest["inputs"]["holdout_labels_read"] is False
    assert dataset.manifest["inputs"]["round2_included"] is False
    assert dataset.manifest["split_counts"] == {"development": 1, "train": 5}
    assert dataset.manifest["label_counts"] == {
        "development:1": 1,
        "train:0": 4,
        "train:1": 1,
    }
    assert dataset.manifest["gold_entity_counts"] == {
        "development:TRIỆU_CHỨNG": 1,
        "train:KẾT_QUẢ_XÉT_NGHIỆM": 1,
        "train:TRIỆU_CHỨNG": 2,
    }

    output = tmp_path / "output"
    write_phase1_proposal_dataset(dataset, output)
    first = (output / "examples.jsonl").read_bytes()
    assert hashlib.sha256(first).hexdigest() == dataset.manifest["examples_sha256"]
    write_phase1_proposal_dataset(dataset, output)
    assert (output / "examples.jsonl").read_bytes() == first


def test_proposal_dataset_rejects_holdout_fingerprint_drift(tmp_path: Path) -> None:
    paths = _write_dataset_fixture(tmp_path)
    model_manifest = json.loads(paths["model_manifest"].read_text(encoding="utf-8"))
    model_manifest["excluded_holdout"]["document_ids_sha256"] = "0" * 64
    paths["model_manifest"].write_text(
        json.dumps(model_manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="excluded-holdout fingerprint"):
        build_phase1_proposal_dataset(
            paths["matrix"],
            paths["input"],
            paths["gold"],
            paths["model_manifest"],
            paths["holdout_manifest"],
            source_roles={
                "pipeline": ProposalSourceRole.RULE,
                "qwen": ProposalSourceRole.LLM,
            },
        )


def test_final_fit_governance_reads_all_reviewed_manual_gold(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    gold_dir = tmp_path / "gold"
    input_dir.mkdir()
    gold_dir.mkdir()
    assignments: list[dict[str, str]] = []
    matrix_rows: list[dict[str, object]] = []
    document_ids = [str(index) for index in range(1, 101)]
    for document_id in document_ids:
        text = "sốt"
        document_path = input_dir / f"{document_id}.txt"
        gold_path = gold_dir / f"{document_id}.json"
        document_path.write_text(text, encoding="utf-8")
        gold_path.write_text(
            json.dumps([_entity(text, "TRIỆU_CHỨNG", 0)], ensure_ascii=False),
            encoding="utf-8",
        )
        assignments.append(
            {
                "document_id": document_id,
                "split": "train" if int(document_id) <= 80 else "holdout",
                "document_sha256": _sha256(document_path),
                "gold_sha256": _sha256(gold_path),
            }
        )
        matrix_rows.append(
            _proposal(
                text,
                "TRIỆU_CHỨNG",
                0,
                sources=["pipeline"],
                status="source_only",
                proposal_id=f"proposal-{document_id}",
                document_id=document_id,
            )
        )

    train_ids = document_ids[:80]
    holdout_ids = document_ids[80:]
    holdout_manifest = {
        "schema_version": "phase1-manual-gold-split.v1",
        "corpus": {"fingerprint_sha256": "corpus-v1"},
        "splits": {
            "train": {"document_ids": train_ids},
            "holdout": {"document_ids": holdout_ids},
        },
        "assignments": assignments,
    }
    holdout_path = tmp_path / "holdout.json"
    holdout_path.write_text(json.dumps(holdout_manifest), encoding="utf-8")
    model_manifest = {
        "schema_version": "phase1-model-training-split.v1",
        "source_corpus_fingerprint_sha256": "corpus-v1",
        "round2_included": False,
        "source_document_ids": {
            "train": train_ids[:60],
            "development": train_ids[60:],
        },
        "excluded_holdout": {
            "document_count": len(holdout_ids),
            "document_ids_sha256": hashlib.sha256(
                "\n".join(sorted(holdout_ids, key=int)).encode("utf-8")
            ).hexdigest(),
        },
    }
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(model_manifest), encoding="utf-8")
    matrix_path = tmp_path / "matrix.jsonl"
    matrix_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in matrix_rows
        ),
        encoding="utf-8",
    )

    dataset = build_phase1_proposal_dataset(
        matrix_path,
        input_dir,
        gold_dir,
        model_path,
        holdout_path,
        source_roles={"pipeline": ProposalSourceRole.RULE},
        training_governance_path=(
            "configs/models/phase1-training-governance-2026-07-30.yaml"
        ),
    )

    assert len({example.document_id for example in dataset.examples}) == 100
    assert dataset.manifest["inputs"]["holdout_labels_read"] is True
    assert dataset.manifest["decision_authority"]["auto_promote"] is False


def _write_dataset_fixture(tmp_path: Path) -> dict[str, Path]:
    input_dir = tmp_path / "input"
    gold_dir = tmp_path / "gold"
    input_dir.mkdir()
    gold_dir.mkdir()
    documents = {
        "1": "đau ngực ho 120 mg",
        "2": "Sốt",
    }
    gold = {
        "1": [
            _entity("đau ngực", "TRIỆU_CHỨNG", 0),
            _entity("ho", "TRIỆU_CHỨNG", 9),
            _entity("120", "KẾT_QUẢ_XÉT_NGHIỆM", 12),
        ],
        "2": [_entity("Sốt", "TRIỆU_CHỨNG", 0)],
    }
    assignments = []
    for document_id in ("1", "2"):
        document_path = input_dir / f"{document_id}.txt"
        gold_path = gold_dir / f"{document_id}.json"
        document_path.write_text(documents[document_id], encoding="utf-8")
        gold_path.write_text(json.dumps(gold[document_id], ensure_ascii=False), encoding="utf-8")
        assignments.append(
            {
                "document_id": document_id,
                "split": "train",
                "document_sha256": _sha256(document_path),
                "gold_sha256": _sha256(gold_path),
            }
        )
    # No holdout source or label file is created. A successful build proves the builder did not
    # open holdout content after validating the frozen id contract.
    assignments.append(
        {
            "document_id": "3",
            "split": "holdout",
            "document_sha256": "a" * 64,
            "gold_sha256": "b" * 64,
        }
    )

    holdout_manifest = {
        "schema_version": "phase1-manual-gold-split.v1",
        "corpus": {"fingerprint_sha256": "corpus-v1"},
        "splits": {
            "train": {"document_ids": ["1", "2"]},
            "holdout": {"document_ids": ["3"]},
        },
        "assignments": assignments,
    }
    holdout_path = tmp_path / "holdout.json"
    holdout_path.write_text(json.dumps(holdout_manifest), encoding="utf-8")
    model_manifest = {
        "schema_version": "phase1-model-training-split.v1",
        "source_corpus_fingerprint_sha256": "corpus-v1",
        "round2_included": False,
        "source_document_ids": {
            "train": ["1"],
            "development": ["2"],
        },
        "excluded_holdout": {
            "document_count": 1,
            "document_ids_sha256": hashlib.sha256(b"3").hexdigest(),
        },
    }
    model_path = tmp_path / "model-split.json"
    model_path.write_text(json.dumps(model_manifest), encoding="utf-8")

    rows = [
        _proposal(
            "đau ngực",
            "TRIỆU_CHỨNG",
            0,
            sources=["pipeline", "qwen"],
            status="exact_agreement",
            proposal_id="exact",
            document_id="1",
        ),
        _proposal(
            "đau ngực",
            "CHẨN_ĐOÁN",
            0,
            sources=["pipeline"],
            status="type_conflict",
            proposal_id="type",
            document_id="1",
        ),
        _proposal(
            "ngực",
            "TRIỆU_CHỨNG",
            4,
            sources=["pipeline"],
            status="overlap_agreement",
            proposal_id="boundary",
            document_id="1",
        ),
        _proposal(
            "ngực ho",
            "CHẨN_ĐOÁN",
            4,
            sources=["qwen"],
            status="source_only",
            proposal_id="boundary-type",
            document_id="1",
        ),
        _proposal(
            "mg",
            "THUỐC",
            16,
            sources=["qwen"],
            status="source_only",
            proposal_id="spurious",
            document_id="1",
        ),
        _proposal(
            "Sốt",
            "TRIỆU_CHỨNG",
            0,
            sources=["qwen"],
            status="source_only",
            proposal_id="dev-exact",
            document_id="2",
        ),
        _proposal(
            "holdout",
            "TRIỆU_CHỨNG",
            0,
            sources=["qwen"],
            status="source_only",
            proposal_id="sealed",
            document_id="3",
        ),
    ]
    matrix_path = tmp_path / "matrix.jsonl"
    matrix_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "matrix": matrix_path,
        "input": input_dir,
        "gold": gold_dir,
        "model_manifest": model_path,
        "holdout_manifest": holdout_path,
    }


def _proposal(
    text: str,
    entity_type: str,
    start: int,
    *,
    sources: list[str],
    status: str,
    proposal_id: str = "proposal",
    document_id: str = "1",
) -> dict[str, object]:
    return {
        "document_id": document_id,
        "proposal_id": proposal_id,
        "text": text,
        "type": entity_type,
        "position": [start, start + len(text)],
        "sources": sources,
        "source_count": len(sources),
        "all_source_agreement": len(sources) == 2,
        "status": status,
    }


def _entity(text: str, entity_type: str, start: int) -> dict[str, object]:
    return {
        "text": text,
        "type": entity_type,
        "position": [start, start + len(text)],
        "assertions": [],
        "candidates": [],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
