"""Contract checks for the redistributable product benchmark fixture."""

import json
from pathlib import Path

import pytest
import yaml

from clingrounder.evaluation.dataset_benchmark import _load_examples


ROOT = Path("benchmarks/vi_clinical_grounding_v1")


def test_manifest_and_split_paths_are_self_contained() -> None:
    manifest = yaml.safe_load((ROOT / "dataset_manifest.yaml").read_text())
    assert manifest["schema_version"] == "clingrounder.dataset-manifest.v1"
    assert manifest["dataset"]["status"] == "synthetic_pilot"
    assert manifest["review"]["status"] == "pending"
    assert manifest["review"]["reviewers_required"] == 2
    for split in manifest["splits"].values():
        assert (ROOT / split["path"]).is_file()
        assert len(split["sha256"]) == 64


def test_schema_is_valid_json_and_declares_fixed_span_contract() -> None:
    schema = json.loads((ROOT / "schema.json").read_text())
    span = schema["properties"]["entities"]["items"]["properties"]["span"]
    assert span["items"] is False
    assert span["minItems"] == span["maxItems"] == 2
    assert len(span["prefixItems"]) == 2


def test_fixture_offsets_and_ids_are_valid() -> None:
    seen_ids: set[str] = set()
    for line in (ROOT / "data" / "test.jsonl").read_text().splitlines():
        record = json.loads(line)
        assert record["document_id"] not in seen_ids
        seen_ids.add(record["document_id"])
        for entity in record["entities"]:
            start, end = entity["span"]
            assert 0 <= start < end <= len(record["text"])
            assert record["text"][start:end] == entity["text"]


@pytest.mark.parametrize(
    ("relation", "message"),
    [
        ({"id": "r1", "head": "e1", "tail": "missing", "type": "HAS_VALUE"}, "unknown entity"),
        ({"id": "r1", "head": "e1", "tail": "e1", "type": "HAS_VALUE"}, "self-loop"),
        ({"id": "r1", "head": "e1", "tail": "e2", "type": "NOT_A_RELATION"}, "unsupported relation"),
    ],
)
def test_relation_contract_rejects_invalid_records(
    tmp_path: Path, relation: dict[str, str], message: str
) -> None:
    row = {
        "document_id": "doc-1",
        "text": "Sốt.",
        "entities": [
            {
                "id": "e1",
                "span": [0, 3],
                "text": "Sốt",
                "type": "SYMPTOM",
                "assertion": "PRESENT",
                "code_system": "NONE",
                "code": None,
            },
            {
                "id": "e2",
                "span": [0, 3],
                "text": "Sốt",
                "type": "SYMPTOM",
                "assertion": "PRESENT",
                "code_system": "NONE",
                "code": None,
            },
        ],
        "relations": [relation],
    }
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match=message):
        _load_examples(path)


def test_relation_contract_rejects_duplicate_relation_ids(tmp_path: Path) -> None:
    row = {
        "document_id": "doc-1",
        "text": "Sốt.",
        "entities": [
            {
                "id": "e1",
                "span": [0, 3],
                "text": "Sốt",
                "type": "SYMPTOM",
                "assertion": "PRESENT",
                "code_system": "NONE",
                "code": None,
            },
            {
                "id": "e2",
                "span": [0, 3],
                "text": "Sốt",
                "type": "SYMPTOM",
                "assertion": "PRESENT",
                "code_system": "NONE",
                "code": None,
            },
        ],
        "relations": [
            {"id": "r1", "head": "e1", "tail": "e2", "type": "ASSOCIATED_WITH"},
            {"id": "r1", "head": "e1", "tail": "e2", "type": "ASSOCIATED_WITH"},
        ],
    }
    path = tmp_path / "duplicate.jsonl"
    path.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="duplicate relation id"):
        _load_examples(path)


def test_entity_contract_rejects_inconsistent_code_system(tmp_path: Path) -> None:
    row = {
        "document_id": "doc-1",
        "text": "Sốt.",
        "entities": [
            {
                "id": "e1",
                "span": [0, 3],
                "text": "Sốt",
                "type": "SYMPTOM",
                "assertion": "PRESENT",
                "code_system": "NONE",
                "code": "fabricated",
            }
        ],
        "relations": [],
    }
    path = tmp_path / "invalid-code.jsonl"
    path.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="NONE code system"):
        _load_examples(path)
