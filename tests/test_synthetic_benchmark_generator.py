import json
from pathlib import Path

from scripts.generate_vi_clinical_benchmark import generate_snapshot


def test_synthetic_snapshot_is_deterministic_and_offset_safe(tmp_path: Path) -> None:
    first = generate_snapshot(
        tmp_path / "first",
        train_documents=4,
        validation_documents=2,
        test_documents=3,
        seed=42,
    )
    second = generate_snapshot(
        tmp_path / "second",
        train_documents=4,
        validation_documents=2,
        test_documents=3,
        seed=42,
    )

    assert first["splits"] == second["splits"]
    for split in ("train", "validation", "test"):
        first_rows = (tmp_path / "first" / f"{split}.jsonl").read_bytes()
        second_rows = (tmp_path / "second" / f"{split}.jsonl").read_bytes()
        assert first_rows == second_rows
        for line in first_rows.decode().splitlines():
            record = json.loads(line)
            for entity in record["entities"]:
                start, end = entity["span"]
                assert record["text"][start:end] == entity["text"]


def test_synthetic_snapshot_keeps_template_groups_disjoint(tmp_path: Path) -> None:
    manifest = generate_snapshot(
        tmp_path,
        train_documents=2,
        validation_documents=2,
        test_documents=2,
    )

    groups = [set(split["template_groups"]) for split in manifest["splits"].values()]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert manifest["dataset"]["human_reviewed"] is False
