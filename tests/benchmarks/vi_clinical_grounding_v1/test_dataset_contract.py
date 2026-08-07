"""Contract checks for the redistributable product benchmark fixture."""

import json
from pathlib import Path

import yaml


ROOT = Path("benchmarks/vi_clinical_grounding_v1")


def test_manifest_and_split_paths_are_self_contained() -> None:
    manifest = yaml.safe_load((ROOT / "dataset_manifest.yaml").read_text())
    assert manifest["schema_version"] == "clingrounder.dataset-manifest.v1"
    assert manifest["dataset"]["status"] == "synthetic_pilot"
    for split in manifest["splits"].values():
        assert (ROOT / split["path"]).is_file()


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

