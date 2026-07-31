"""Authorized GT materialization keeps organizer LF offsets exact and reproducible."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from medical_kg_nlp.benchmarks.phase1.authorized_ground_truth import (
    load_phase1_authorized_ground_truth,
    materialize_phase1_authorized_ground_truth,
)


def test_authorized_ground_truth_uses_lf_child_offsets_and_writes_manifest(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "part2.zip")
    governance = _governance(tmp_path / "governance.yaml", archive)

    corpus = load_phase1_authorized_ground_truth(governance, archive_path=archive)
    manifest = materialize_phase1_authorized_ground_truth(corpus, tmp_path / "materialized")

    assert corpus.source_texts["authorized_gt:1"] == "đau\nsốt"
    assert corpus.gold_rows["authorized_gt:1"][1]["position"] == [4, 7]
    assert manifest["document_count"] == 100
    documents = (tmp_path / "materialized" / "documents.jsonl").read_text(encoding="utf-8")
    assert "\\r" not in documents


def test_authorized_ground_truth_rejects_offsets_against_original_crlf_view(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "part2.zip", invalid_lf_offset=True)
    governance = _governance(tmp_path / "governance.yaml", archive)

    try:
        load_phase1_authorized_ground_truth(governance, archive_path=archive)
    except ValueError as error:
        assert "LF raw-offset invariant" in str(error)
    else:
        raise AssertionError("Expected CRLF-coordinate annotation to be rejected")


def _archive(path: Path, *, invalid_lf_offset: bool = False) -> Path:
    input_payload = io.BytesIO()
    gt_payload = io.BytesIO()
    with ZipFile(input_payload, "w", ZIP_DEFLATED) as input_zip, ZipFile(
        gt_payload,
        "w",
        ZIP_DEFLATED,
    ) as gt_zip:
        for index in range(1, 101):
            text = "đau\r\nsốt"
            input_zip.writestr(f"input/{index}.txt", text)
            end = 8 if invalid_lf_offset and index == 1 else 7
            gt_zip.writestr(
                f"output/{index}.json",
                json.dumps(
                    [
                        {"text": "đau", "type": "TRIỆU_CHỨNG", "position": [0, 3]},
                        {"text": "sốt", "type": "TRIỆU_CHỨNG", "position": [4, end]},
                    ],
                    ensure_ascii=False,
                ),
            )
    with ZipFile(path, "w", ZIP_DEFLATED) as parent:
        parent.writestr("input.zip", input_payload.getvalue())
        parent.writestr("gt.zip", gt_payload.getvalue())
    return path


def _governance(path: Path, archive: Path) -> Path:
    with ZipFile(archive) as parent:
        input_hash = hashlib.sha256(parent.read("input.zip")).hexdigest()
        gt_hash = hashlib.sha256(parent.read("gt.zip")).hexdigest()
    path.write_text(
        "\n".join(
            (
                "schema_version: phase1-training-governance.v1",
                'effective_from: "2026-07-30"',
                "manual_gold:",
                "  path: data/manual_gold",
                "  source_documents: data/raw/input",
                "  expected_document_count: 100",
                "  usage: train_all",
                "  legacy_split_role: diagnostic_only",
                "authorized_ground_truth:",
                "  source_id: phase1_part2_leaked_bundle",
                "  archive_env: PHASE1_PART2_ARCHIVE",
                f"  archive_sha256: {hashlib.sha256(archive.read_bytes()).hexdigest()}",
                f"  input_zip_sha256: {input_hash}",
                f"  gt_zip_sha256: {gt_hash}",
                "  expected_document_count: 100",
                "  usage: supervised_training",
                "  offset_coordinate_view: crlf_to_lf_child_document",
                "friend31:",
                "  source_id: friend31",
                "  role: distillation_reference_only",
                "  runtime_source_allowed: false",
                "  submission_seed_allowed: false",
                "decision_authority:",
                "  local_metrics: diagnostic_only",
                "  local_can_promote: false",
                "  local_can_reject: false",
                "  official_submission: sole_promotion_and_rejection_authority",
                "  hard_validation_can_block_packaging: true",
                "  major_change_requires_submission_artifact: true",
                "  major_change_may_close_without_artifact: false",
                "reproducibility:",
                "  require_repository_owned_inference: true",
                "  require_pinned_checkpoint: true",
                "  require_pinned_config: true",
                "  require_source_fingerprints: true",
            )
        ),
        encoding="utf-8",
    )
    return path
