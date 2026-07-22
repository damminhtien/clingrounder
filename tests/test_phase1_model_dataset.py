"""Five-type Phase 1 model data must preserve offsets and frozen split isolation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from medical_kg_nlp.benchmarks.phase1.manual_gold import (
    build_manual_gold_split_manifest,
    write_manual_gold_split_manifest,
)
from medical_kg_nlp.benchmarks.phase1.model_dataset import (
    PHASE1_FIVE_TYPE_LABELS,
    Phase1ModelDatasetConfig,
    build_phase1_model_dataset,
    build_phase1_model_splits,
)
from medical_kg_nlp.cli.main import main
from medical_kg_nlp.mining.records import (
    AccessClass,
    MinedDocument,
    RedistributionPolicy,
)


def test_model_split_keeps_duplicate_groups_together() -> None:
    documents = [_document("duplicate-a", "same raw medical note")]
    documents.append(_document("duplicate-b", "same raw medical note"))
    for index in range(40):
        tokens = " ".join(
            hashlib.sha256(f"{index}:{token}".encode()).hexdigest()[:16]
            for token in range(40)
        )
        documents.append(_document(f"unique-{index}", tokens))

    first = build_phase1_model_splits(
        documents,
        development_fraction=0.2,
        split_salt="42",
    )
    second = build_phase1_model_splits(
        tuple(reversed(documents)),
        development_fraction=0.2,
        split_salt="42",
    )

    splits, groups, group_counts = first
    assert first == second
    assert splits["duplicate-a"] == splits["duplicate-b"]
    assert groups["duplicate-a"] == groups["duplicate-b"]
    assert set(splits.values()) == {"train", "development"}
    assert group_counts["raw_exact"] == 1


def test_model_dataset_excludes_frozen_holdout_and_round2(
    tmp_path: Path,
    capsys,
) -> None:
    input_dir = tmp_path / "input"
    gold_dir = tmp_path / "gold"
    input_dir.mkdir()
    gold_dir.mkdir()
    for source_id in range(1, 41):
        text, rows = _five_type_note(source_id)
        (input_dir / f"{source_id}.txt").write_text(text, encoding="utf-8")
        (gold_dir / f"{source_id}.json").write_text(
            json.dumps(rows, ensure_ascii=False),
            encoding="utf-8",
        )
    source_manifest = build_manual_gold_split_manifest(gold_dir, input_dir)
    source_manifest_path = tmp_path / "holdout_manifest.json"
    write_manual_gold_split_manifest(source_manifest, source_manifest_path)
    output_dir = tmp_path / "five-type"

    config = Phase1ModelDatasetConfig(
        input_dir=input_dir,
        gold_dir=gold_dir,
        frozen_split_manifest=source_manifest_path,
        development_fraction=0.2,
        split_salt="42",
        max_characters=256,
    )
    assert (
        main(
            [
                "benchmark",
                "phase1",
                "model-data",
                "build",
                "--input-dir",
                str(input_dir),
                "--gold-dir",
                str(gold_dir),
                "--frozen-split-manifest",
                str(source_manifest_path),
                "--output-dir",
                str(output_dir),
                "--development-fraction",
                "0.2",
                "--split-salt",
                "42",
                "--max-characters",
                "256",
            ]
        )
        == 0
    )
    first = json.loads(capsys.readouterr().out)
    second = build_phase1_model_dataset(output_dir, config=config)

    assert first == second
    frozen_train = {
        row["document_id"]
        for row in source_manifest["assignments"]
        if row["split"] == "train"
    }
    frozen_holdout = {
        row["document_id"]
        for row in source_manifest["assignments"]
        if row["split"] == "holdout"
    }
    assert first["dataset"]["document_count"] == len(frozen_train)
    assert first["dataset"]["annotation_count"] == len(frozen_train) * 5
    assert set(first["dataset"]["entity_type_counts"]) == set(
        PHASE1_FIVE_TYPE_LABELS
    )
    assert first["build_contract"]["round2_included"] is False
    assert (
        first["build_contract"]["public_executable_spec"]["included_in_training"]
        is False
    )

    model_split = json.loads((output_dir / "split_manifest.json").read_text())
    selected_source_ids = {
        source_id
        for values in model_split["source_document_ids"].values()
        for source_id in values
    }
    assert selected_source_ids == frozen_train
    assert selected_source_ids.isdisjoint(frozen_holdout)
    assert set(model_split["splits"].values()) == {"train", "development"}
    assert model_split["excluded_holdout"]["document_count"] == len(frozen_holdout)
    assert model_split["round2_included"] is False

    span_manifest = json.loads((output_dir / "manifest.json").read_text())
    assert span_manifest["output"] == "spans.jsonl"
    assert span_manifest["inputs"]["documents"]["path"] == "documents.jsonl"
    assert span_manifest["inputs"]["annotations"]["path"] == "annotations.jsonl"
    assert span_manifest["inputs"]["split_manifest"]["path"] == "split_manifest.json"


def _document(document_id: str, text: str) -> MinedDocument:
    return MinedDocument(
        document_id=document_id,
        text=text,
        language="vi",
        note_type="clinical_note",
        source_artifact_id="phase1-manual-gold:test",
        access_class=AccessClass.LOCAL_PRIVATE,
        redistribution=RedistributionPolicy.PROHIBITED,
        hosted_processing_allowed=False,
    )


def _five_type_note(source_id: int) -> tuple[str, list[dict[str, object]]]:
    mentions = (
        (f"triệu chứng {source_id}", "TRIỆU_CHỨNG"),
        (f"xét nghiệm {source_id}", "TÊN_XÉT_NGHIỆM"),
        (f"kết quả {source_id}", "KẾT_QUẢ_XÉT_NGHIỆM"),
        (f"chẩn đoán {source_id}", "CHẨN_ĐOÁN"),
        (f"thuốc {source_id}", "THUỐC"),
    )
    noise = " ".join(
        hashlib.sha256(f"note:{source_id}:{index}".encode()).hexdigest()[:16]
        for index in range(60)
    )
    text = " | ".join(mention for mention, _ in mentions) + "\n" + noise
    rows: list[dict[str, object]] = []
    cursor = 0
    for mention, entity_type in mentions:
        start = text.index(mention, cursor)
        end = start + len(mention)
        rows.append(
            {
                "text": mention,
                "position": [start, end],
                "type": entity_type,
                "assertions": [],
                "candidates": [],
            }
        )
        cursor = end
    return text, rows
