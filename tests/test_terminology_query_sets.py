"""Tests for deterministic benchmark queries derived from mined aliases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clingrounder.terminology import (
    build_alias_overlay_queries,
    build_linked_proposal_queries,
    load_terminology_queries,
    write_alias_overlay_query_set,
    write_linked_proposal_query_set,
)


def _write_overlay(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_alias_overlay_queries_group_normalized_surfaces_by_semantic_space(
    tmp_path: Path,
) -> None:
    overlay = tmp_path / "aliases.jsonl"
    _write_overlay(
        overlay,
        [
            {
                "alias": "Tăng huyết áp",
                "code": "I10",
                "code_system": "ICD-10",
                "semantic_type": "DISEASE",
            },
            {
                "alias": "tăng huyết áp",
                "code": "I11",
                "code_system": "ICD-10",
                "semantic_type": "DISEASE",
            },
            {
                "alias": "Tăng huyết áp",
                "code": "123",
                "code_system": "RxNorm",
                "semantic_type": "DRUG",
            },
        ],
    )

    queries = build_alias_overlay_queries((overlay,))

    assert len(queries) == 2
    disease = next(query for query in queries if query.entity_type.value == "DISEASE")
    assert disease.expected_codes == ("I10", "I11")
    assert disease.mention == "Tăng huyết áp"


def test_query_set_writer_pins_inputs_and_round_trips(tmp_path: Path) -> None:
    overlay = tmp_path / "aliases.jsonl"
    output = tmp_path / "queries.jsonl"
    manifest_path = tmp_path / "manifest.json"
    _write_overlay(
        overlay,
        [
            {
                "alias": "Salmonella",
                "code": "A02.9",
                "code_system": "ICD-10",
                "semantic_type": "DISEASE",
            }
        ],
    )

    manifest = write_alias_overlay_query_set((overlay,), output, manifest_path)

    assert manifest["query_count"] == 1
    assert manifest["ambiguous_query_count"] == 0
    assert manifest["sources"][0]["sha256"]
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert load_terminology_queries(output)[0].expected_codes == ("A02.9",)


def test_alias_overlay_queries_reject_unlinked_recognition_alias(tmp_path: Path) -> None:
    overlay = tmp_path / "aliases.jsonl"
    _write_overlay(
        overlay,
        [
            {
                "alias": "đau đầu",
                "code": "NONE:abc",
                "code_system": "NONE",
                "semantic_type": "SYMPTOM",
            }
        ],
    )

    with pytest.raises(ValueError, match="cannot use code system NONE"):
        build_alias_overlay_queries((overlay,))


def test_linked_proposal_queries_report_reference_leakage_slices(tmp_path: Path) -> None:
    train_overlay = tmp_path / "train_aliases.jsonl"
    proposals = tmp_path / "development_proposals.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    manifest_path = tmp_path / "manifest.json"
    _write_overlay(
        train_overlay,
        [
            {
                "alias": "tăng huyết áp",
                "code": "I10",
                "code_system": "ICD-10",
                "semantic_type": "DISEASE",
            }
        ],
    )
    _write_overlay(
        proposals,
        [
            {
                "normalized_alias": "tăng huyết áp",
                "code": "I10",
                "code_system": "ICD-10",
                "semantic_type": "DISEASE",
                "surface_variants": [{"surface": "Tăng huyết áp"}],
            },
            {
                "normalized_alias": "cao huyết áp",
                "code": "I10",
                "code_system": "ICD-10",
                "semantic_type": "DISEASE",
                "surface_variants": [{"surface": "Cao huyết áp"}],
            },
            {
                "normalized_alias": "bệnh mới",
                "code": "I11",
                "code_system": "ICD-10",
                "semantic_type": "DISEASE",
                "surface_variants": [{"surface": "Bệnh mới"}],
            },
        ],
    )

    queries = build_linked_proposal_queries(
        (proposals,),
        reference_overlay_paths=(train_overlay,),
    )
    manifest = write_linked_proposal_query_set(
        (proposals,),
        queries_path,
        manifest_path,
        reference_overlay_paths=(train_overlay,),
    )

    by_mention = {query.mention: query for query in queries}
    assert by_mention["Tăng huyết áp"].slices == (
        "alias_seen_in_reference",
        "code_seen_in_reference",
    )
    assert by_mention["Cao huyết áp"].slices == (
        "alias_unseen_in_reference",
        "code_seen_in_reference",
    )
    assert by_mention["Bệnh mới"].slices == (
        "alias_unseen_in_reference",
        "code_unseen_in_reference",
    )
    assert manifest["slice_counts"]["alias_unseen_in_reference"] == 2
    assert load_terminology_queries(queries_path) == queries
