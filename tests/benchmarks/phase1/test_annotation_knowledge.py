import json
import subprocess
import sys
from pathlib import Path

import pytest

import yaml

from medical_kg_nlp.benchmarks.phase1.annotation_knowledge import (
    compile_annotation_knowledge,
    write_annotation_knowledge,
)


def test_compiler_builds_strict_alias_and_exclusion_without_document_rules(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    source1 = tmp_path / "1.txt"
    source2 = tmp_path / "2.txt"
    source1.write_text("ho phẫu thuật", encoding="utf-8")
    source2.write_text("ho", encoding="utf-8")
    _write_gold(gold_dir / "1.json", [_row("ho", "TRIỆU_CHỨNG", 0)])
    _write_gold(gold_dir / "2.json", [_row("ho", "TRIỆU_CHỨNG", 0)])
    procedure_start = source1.read_text(encoding="utf-8").index("phẫu thuật")
    manifest = tmp_path / "review_manifest.jsonl"
    _write_manifest(
        manifest,
        [
            _manifest_row(
                "1",
                gold_dir / "1.json",
                source1,
                review_candidates=[
                    {
                        "text": "phẫu thuật",
                        "position": [procedure_start, procedure_start + len("phẫu thuật")],
                        "reason": "Surgical procedure; schema has no procedure type.",
                    }
                ],
                guideline_notes=["Each occurrence is a separate span-level entity."],
            ),
            _manifest_row("2", gold_dir / "2.json", source2),
        ],
    )

    report = compile_annotation_knowledge(gold_dir=gold_dir, manifest_path=manifest)
    output_dir = tmp_path / "compiled"
    write_annotation_knowledge(report, output_dir)

    policy = report["policy"]
    assert policy["aliases"]["strict"]["TRIỆU_CHỨNG"] == ["ho"]
    assert policy["exclusions"]["strict"]["procedure_or_device"] == ["phẫu thuật"]
    assert "document_id" not in json.dumps(policy, ensure_ascii=False)
    assert report["summary"]["reviewed_document_count"] == 2
    assert report["summary"]["accepted_entity_count"] == 2
    assert (output_dir / "annotation_knowledge.json").exists()
    assert (output_dir / "policy_conflicts.csv").exists()
    assert yaml.safe_load((output_dir / "phase1_annotation_policy.yaml").read_text(encoding="utf-8"))[
        "runtime_constraints"
    ]["document_specific_rules"] is False


def test_compiler_reports_type_positive_negative_offset_and_count_conflicts(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    source1 = tmp_path / "1.txt"
    source2 = tmp_path / "2.txt"
    source1.write_text("đau", encoding="utf-8")
    source2.write_text("đau", encoding="utf-8")
    _write_gold(gold_dir / "1.json", [_row("đau", "TRIỆU_CHỨNG", 0)])
    _write_gold(gold_dir / "2.json", [_row("đau", "CHẨN_ĐOÁN", 0)])
    manifest = tmp_path / "review_manifest.jsonl"
    first = _manifest_row(
        "1",
        gold_dir / "1.json",
        source1,
        review_candidates=[
            {
                "text": "đau",
                "position": [0, 2],
                "reason": "Generic phrase; review separately.",
            }
        ],
    )
    first["entity_count"] = 2
    _write_manifest(manifest, [first, _manifest_row("2", gold_dir / "2.json", source2)])

    report = compile_annotation_knowledge(gold_dir=gold_dir, manifest_path=manifest)

    conflict_types = {row["conflict_type"] for row in report["conflicts"]}
    assert "manifest_entity_count_mismatch" in conflict_types
    assert "positive_negative_same_mention" in conflict_types
    assert "positive_type_disagreement" in conflict_types
    assert "review_offset_mismatch" in conflict_types
    assert "đau" in report["policy"]["unstable_mentions"]
    assert "đau" not in report["policy"]["aliases"]["strict"]["TRIỆU_CHỨNG"]
    assert "đau" not in report["policy"]["aliases"]["strict"]["CHẨN_ĐOÁN"]


def test_compiler_records_concept_level_conflict_resolution(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    source = tmp_path / "1.txt"
    source.write_text("ho", encoding="utf-8")
    _write_gold(gold_dir / "1.json", [_row("ho", "TRIỆU_CHỨNG", 0)])
    manifest = tmp_path / "review_manifest.jsonl"
    _write_manifest(
        manifest,
        [
            _manifest_row(
                "1",
                gold_dir / "1.json",
                source,
                review_candidates=[
                    {
                        "text": "ho",
                        "position": [0, 2],
                        "reason": "Generic short alias requires context.",
                    }
                ],
            )
        ],
    )
    decisions = tmp_path / "conflict_decisions.jsonl"
    decisions.write_text(
        json.dumps(
            {
                "conflict_type": "positive_negative_same_mention",
                "normalized_text": "ho",
                "action": "context_required",
                "reason": "Accept only as a standalone symptom token.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = compile_annotation_knowledge(
        gold_dir=gold_dir,
        manifest_path=manifest,
        conflict_decisions_path=decisions,
    )

    assert report["conflicts"] == []
    assert report["summary"]["resolved_conflict_count"] == 1
    assert report["conflict_resolutions"][0]["resolution_action"] == "context_required"
    assert "ho" in report["policy"]["unstable_mentions"]


def test_compiler_keeps_candidate_mapping_notes_out_of_entity_conflicts(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    source = tmp_path / "1.txt"
    source.write_text("insulin", encoding="utf-8")
    _write_gold(gold_dir / "1.json", [_row("insulin", "THUỐC", 0)])
    manifest = tmp_path / "review_manifest.jsonl"
    _write_manifest(
        manifest,
        [
            _manifest_row(
                "1",
                gold_dir / "1.json",
                source,
                review_candidates=[
                    {
                        "text": "insulin",
                        "position": [0, len("insulin")],
                        "scope": "candidate_mapping",
                        "reason": "Entity retained, but no exact product candidate is justified.",
                    }
                ],
            )
        ],
    )

    report = compile_annotation_knowledge(gold_dir=gold_dir, manifest_path=manifest)

    assert report["conflicts"] == []
    assert report["knowledge"]["rejected_mentions"] == []


def test_compiler_records_confirmed_runtime_exclusion_resolution(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    source = tmp_path / "1.txt"
    source.write_text("nghi ngờ", encoding="utf-8")
    _write_gold(gold_dir / "1.json", [])
    manifest = tmp_path / "review_manifest.jsonl"
    _write_manifest(
        manifest,
        [
            _manifest_row(
                "1",
                gold_dir / "1.json",
                source,
                review_candidates=[
                    {
                        "text": "nghi ngờ",
                        "position": [0, len("nghi ngờ")],
                        "reason": "Uncertainty cue; review separately before promotion.",
                    }
                ],
            )
        ],
    )
    decisions = tmp_path / "conflict_decisions.jsonl"
    decisions.write_text(
        json.dumps(
            {
                "conflict_type": "unstable_policy_evidence",
                "normalized_text": "nghi ngờ",
                "action": "exclude_from_runtime",
                "reason": "The uncertainty cue is not a Phase 1 entity.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = compile_annotation_knowledge(
        gold_dir=gold_dir,
        manifest_path=manifest,
        conflict_decisions_path=decisions,
    )

    assert report["conflicts"] == []
    assert report["conflict_resolutions"][0]["resolution_action"] == "exclude_from_runtime"


@pytest.mark.release
def test_annotation_knowledge_cli_smoke(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    source = tmp_path / "1.txt"
    source.write_text("ho", encoding="utf-8")
    _write_gold(gold_dir / "1.json", [_row("ho", "TRIỆU_CHỨNG", 0)])
    manifest = tmp_path / "review_manifest.jsonl"
    _write_manifest(manifest, [_manifest_row("1", gold_dir / "1.json", source)])
    output_dir = tmp_path / "compiled"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/phase1/build_phase1_annotation_knowledge.py",
            "--gold-dir",
            str(gold_dir),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--strict-document-support",
            "1",
            "--split",
            "all",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["reviewed_document_count"] == 1
    assert payload["strict_alias_count"] == 1
    assert (output_dir / "report.md").exists()
    assert (output_dir / "conflict_summary.json").exists()


@pytest.mark.release
def test_annotation_knowledge_cli_fails_on_unresolved_conflict(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    source = tmp_path / "1.txt"
    source.write_text("nghi ngờ", encoding="utf-8")
    _write_gold(gold_dir / "1.json", [])
    manifest = tmp_path / "review_manifest.jsonl"
    _write_manifest(
        manifest,
        [
            _manifest_row(
                "1",
                gold_dir / "1.json",
                source,
                review_candidates=[
                    {
                        "text": "nghi ngờ",
                        "position": [0, len("nghi ngờ")],
                        "reason": "Uncertainty cue; review separately before promotion.",
                    }
                ],
            )
        ],
    )
    decisions = tmp_path / "empty_conflict_decisions.jsonl"
    decisions.write_text("", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/phase1/build_phase1_annotation_knowledge.py",
            "--gold-dir",
            str(gold_dir),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "compiled"),
            "--conflict-decisions",
            str(decisions),
            "--split",
            "all",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "unresolved conflicts" in result.stderr


def test_numeric_lab_result_is_context_required_not_strict(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    manifest_rows: list[dict[str, object]] = []
    for document_id in ("1", "2"):
        source = tmp_path / f"{document_id}.txt"
        source.write_text("17", encoding="utf-8")
        gold = gold_dir / f"{document_id}.json"
        _write_gold(gold, [_row("17", "KẾT_QUẢ_XÉT_NGHIỆM", 0)])
        manifest_rows.append(_manifest_row(document_id, gold, source))
    manifest = tmp_path / "review_manifest.jsonl"
    _write_manifest(manifest, manifest_rows)

    report = compile_annotation_knowledge(gold_dir=gold_dir, manifest_path=manifest)

    aliases = report["policy"]["aliases"]
    assert "17" in aliases["context_required"]["KẾT_QUẢ_XÉT_NGHIỆM"]
    assert "17" not in aliases["strict"]["KẾT_QUẢ_XÉT_NGHIỆM"]


def test_compiler_can_seal_documents_out_of_runtime_policy(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    manifest_rows: list[dict[str, object]] = []
    for document_id, mention in (("1", "ho"), ("2", "sốt")):
        source = tmp_path / f"{document_id}.txt"
        source.write_text(mention, encoding="utf-8")
        gold = gold_dir / f"{document_id}.json"
        _write_gold(gold, [_row(mention, "TRIỆU_CHỨNG", 0)])
        manifest_rows.append(_manifest_row(document_id, gold, source))
    manifest = tmp_path / "review_manifest.jsonl"
    _write_manifest(manifest, manifest_rows)

    report = compile_annotation_knowledge(
        gold_dir=gold_dir,
        manifest_path=manifest,
        strict_document_support=1,
        document_ids={"1"},
    )

    aliases = report["policy"]["aliases"]["strict"]["TRIỆU_CHỨNG"]
    assert aliases == ["ho"]
    assert report["inputs"]["selected_document_ids"] == ["1"]
    assert report["inputs"]["excluded_manifest_document_count"] == 1


def _row(text: str, entity_type: str, start: int) -> dict[str, object]:
    return {
        "text": text,
        "type": entity_type,
        "assertions": [],
        "candidates": [],
        "position": [start, start + len(text)],
    }


def _manifest_row(
    document_id: str,
    gold_file: Path,
    source_file: Path,
    *,
    review_candidates: list[dict[str, object]] | None = None,
    guideline_notes: list[str] | None = None,
) -> dict[str, object]:
    entity_count = len(json.loads(gold_file.read_text(encoding="utf-8")))
    return {
        "document_id": document_id,
        "gold_file": str(gold_file),
        "source_file": str(source_file),
        "status": "strict_v0_reviewed",
        "reviewed_by": "test",
        "review_date": "2026-07-11",
        "entity_count": entity_count,
        "candidate_policy": "test",
        "draft_policy": "test",
        "guideline_notes": guideline_notes or [],
        "review_candidates": review_candidates or [],
    }


def _write_gold(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
