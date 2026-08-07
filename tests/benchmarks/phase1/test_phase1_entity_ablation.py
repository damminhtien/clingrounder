from __future__ import annotations

import hashlib
import json
from pathlib import Path

from clingrounder.benchmarks.phase1.manual_gold import (
    build_manual_gold_split_manifest,
    write_manual_gold_split_manifest,
)
from clingrounder.benchmarks.phase1.phase1_entity_ablation import (
    Phase1EntityAblationConfig,
    run_phase1_entity_ablations,
)


def test_entity_ablation_freezes_split_and_isolates_medication_span(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    gold_dir = tmp_path / "gold"
    base_dir = tmp_path / "base"
    input_dir.mkdir()
    gold_dir.mkdir()
    base_dir.mkdir()

    source_train = "Ghi nhận tăng."
    source_holdout = "amlodipine 10 mg po daily điều trị tăng huyết áp"
    for document_id in range(1, 11):
        source = source_train if document_id == 1 else "Không ghi nhận."
        (input_dir / f"{document_id}.txt").write_text(source, encoding="utf-8")
        _write_rows(gold_dir / f"{document_id}.json", [])
        _write_rows(base_dir / f"{document_id}.json", [])
    (input_dir / "11.txt").write_text(source_holdout, encoding="utf-8")
    _write_rows(
        gold_dir / "11.json",
        [_row("amlodipine 10 mg po daily", "THUỐC", 0)],
    )
    _write_rows(
        base_dir / "1.json",
        [_row("tăng", "KẾT_QUẢ_XÉT_NGHIỆM", source_train.index("tăng"))],
    )
    _write_rows(base_dir / "11.json", [_row("amlodipine", "THUỐC", 0)])

    split_manifest_path = tmp_path / "holdout_manifest.json"
    write_manual_gold_split_manifest(
        build_manual_gold_split_manifest(gold_dir, input_dir),
        split_manifest_path,
    )
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text("schema_version: test\naliases: {}\n", encoding="utf-8")
    dictionary_path = tmp_path / "dictionary.jsonl"
    dictionary_path.write_text("", encoding="utf-8")

    manifest = run_phase1_entity_ablations(
        Phase1EntityAblationConfig(
            base=base_dir,
            expected_base_sha256=_path_sha256(base_dir),
            input_dir=input_dir,
            gold_dir=gold_dir,
            split_manifest=split_manifest_path,
            annotation_policy=policy_path,
            dictionary_paths=(dictionary_path,),
            output_root=tmp_path / "runs",
            journal_dir=tmp_path / "journal",
        )
    )

    variants = {row["name"]: row for row in manifest["variants"]}
    medication = variants["E_MEDICATION_FULL_SPAN"]
    lab = variants["E_LAB_RESULT_RETYPE"]
    assert manifest["split_manifest"]["splits"]["holdout"]["document_ids"] == ["11"]
    assert medication["split_deltas"]["holdout"]["wer_reduction"] > 0
    assert medication["decision"] == "keep_candidate"
    assert lab["split_deltas"]["train"]["spurious_reduction"] == 1
    assert Path(medication["zip"]).exists()
    assert (Path(manifest["run_dir"]) / "summary.csv").exists()


def _row(text: str, entity_type: str, start: int) -> dict[str, object]:
    return {
        "text": text,
        "type": entity_type,
        "assertions": [],
        "candidates": [],
        "position": [start, start + len(text)],
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(child.read_bytes())
    return digest.hexdigest()
