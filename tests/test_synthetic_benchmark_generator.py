import json
from pathlib import Path

from clingrounder.evaluation.dataset_audit import audit_dataset
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


def test_synthetic_snapshot_covers_labs_and_relations(tmp_path: Path) -> None:
    generate_snapshot(tmp_path, train_documents=6, validation_documents=5, test_documents=7)

    records = [
        json.loads(line)
        for line in (tmp_path / "test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    lab_records = [record for record in records if record["relations"]]
    assert lab_records
    for record in lab_records:
        by_id = {entity["id"]: entity for entity in record["entities"]}
        for relation in record["relations"]:
            assert by_id[relation["head"]]["type"] == "LAB_TEST"
            assert by_id[relation["tail"]]["type"] == "LAB_RESULT"
            assert relation["type"] == "HAS_VALUE"


def test_synthetic_snapshot_uses_semantic_roles_and_matching_assertion_cues(
    tmp_path: Path,
) -> None:
    manifest = generate_snapshot(
        tmp_path,
        train_documents=18,
        validation_documents=10,
        test_documents=14,
    )
    records = [
        json.loads(line)
        for split in ("train", "validation", "test")
        for line in (tmp_path / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert set(manifest["assertions"]) == {
        "PRESENT",
        "NEGATED",
        "HISTORICAL",
        "FAMILY",
        "POSSIBLE",
    }
    for record in records:
        template = record["metadata"]["template_group"]
        entities = record["entities"]
        if template.endswith(".medication"):
            assert [entity["type"] for entity in entities] == ["DRUG", "SYMPTOM"]
        if template.endswith(".negation"):
            assert entities[0]["type"] == "SYMPTOM"
            assert entities[0]["assertion"] == "NEGATED"
        if template.endswith(".history"):
            assert all(
                entity["assertion"] == "HISTORICAL"
                for entity in entities
                if entity["type"] == "DISEASE"
            )
        if template.endswith(".family"):
            assert next(
                entity for entity in entities if entity["type"] == "DISEASE"
            )["assertion"] == "FAMILY"
        if template.endswith(".possible"):
            assert next(
                entity for entity in entities if entity["type"] == "DISEASE"
            )["assertion"] == "POSSIBLE"


def test_synthetic_snapshot_is_unique_and_auditable_before_human_review(
    tmp_path: Path,
) -> None:
    manifest = generate_snapshot(tmp_path)

    for split in ("train", "validation", "test"):
        records = [
            json.loads(line)
            for line in (tmp_path / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len({record["text"] for record in records}) == len(records)

    report = audit_dataset(tmp_path)
    assert not report.issues
    assert report.checks["annotation_structure_valid"] is True
    assert report.checks["template_groups_disjoint"] is True
    assert report.checks["normalized_text_splits_disjoint"] is True
    assert report.checks["human_reviewed_release"] is False
    assert report.eligible_for_clinical_claim is False

    assert manifest["dataset"]["version"] == "0.2.0"
