"""Neutral retrieval metrics make terminology promotion measurable."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medical_kg_nlp.terminology import (
    SQLiteTerminologyRepository,
    build_terminology_index,
    evaluate_terminology_queries,
    load_terminology_queries,
)


def test_retrieval_evaluation_detects_alias_coverage_gain(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "rxnorm.jsonl")
    overlay = tmp_path / "aliases.jsonl"
    overlay.write_text(
        json.dumps(
            {
                "alias": "Metformin 500 MG Oral Tablet",
                "target_concept_id": "RX:100",
                "code_system": "RxNorm",
                "code": "100",
                "semantic_type": "DRUG",
                "source": "fixture",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    queries_path = _write_queries(tmp_path / "queries.jsonl")
    queries = load_terminology_queries(queries_path)
    base = build_terminology_index((source,), cache_dir=tmp_path / "base")
    enriched = build_terminology_index(
        (source,),
        alias_overlay_paths=(overlay,),
        cache_dir=tmp_path / "enriched",
    )

    base_repository = SQLiteTerminologyRepository(base.index_path)
    enriched_repository = SQLiteTerminologyRepository(enriched.index_path)
    base_report = evaluate_terminology_queries(base_repository, queries)
    enriched_report = evaluate_terminology_queries(enriched_repository, queries)

    assert base_report["modes"]["exact"]["metrics"]["matched_query_count"] == 1
    assert enriched_report["modes"]["exact"]["metrics"]["matched_query_count"] == 2
    assert enriched_report["modes"]["exact"]["metrics"]["hit_at_1"] == pytest.approx(
        2 / 3
    )
    assert enriched_report["unknown_expected_codes"] == [
        {"code_system": "RxNorm", "code": "999"}
    ]
    assert enriched_report["modes"]["search"]["latency_ms"]["p95"] >= 0.0


def test_query_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = _write_queries(tmp_path / "queries.jsonl")
    path.write_text(path.read_text(encoding="utf-8") * 2, encoding="utf-8")

    with pytest.raises(ValueError, match="IDs must be unique"):
        load_terminology_queries(path)


def test_retrieval_evaluation_reports_query_slice_metrics(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "rxnorm.jsonl")
    queries_path = _write_queries(tmp_path / "queries.jsonl")
    rows = [json.loads(line) for line in queries_path.read_text().splitlines()]
    rows[0]["slices"] = ["alias_unseen_in_reference", "code_seen_in_reference"]
    rows[1]["slices"] = ["alias_seen_in_reference", "code_seen_in_reference"]
    rows[2]["slices"] = ["alias_unseen_in_reference", "code_unseen_in_reference"]
    queries_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = build_terminology_index((source,), cache_dir=tmp_path / "cache")
    repository = SQLiteTerminologyRepository(manifest.index_path)

    report = evaluate_terminology_queries(
        repository,
        load_terminology_queries(queries_path),
    )

    assert report["slice_counts"] == {
        "alias_seen_in_reference": 1,
        "alias_unseen_in_reference": 2,
        "code_seen_in_reference": 2,
        "code_unseen_in_reference": 1,
    }
    exact_slices = report["modes"]["exact"]["slice_metrics"]
    assert exact_slices["alias_seen_in_reference"]["hit_at_1"] == 1.0
    assert exact_slices["alias_unseen_in_reference"]["hit_at_1"] == 0.0


def _write_source(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "concept_id": "RX:100",
                "code": "100",
                "code_system": "RxNorm",
                "canonical_name": "metformin",
                "semantic_type": "DRUG",
                "source": "fixture",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_queries(path: Path) -> Path:
    rows = [
        {
            "query_id": "q1",
            "mention": "Metformin 500 MG Oral Tablet",
            "entity_type": "DRUG",
            "code_system": "RxNorm",
            "expected_codes": ["100"],
        },
        {
            "query_id": "q2",
            "mention": "metformin",
            "entity_type": "DRUG",
            "code_system": "RxNorm",
            "expected_codes": ["100"],
        },
        {
            "query_id": "q3",
            "mention": "unknown drug",
            "entity_type": "DRUG",
            "code_system": "RxNorm",
            "expected_codes": ["999"],
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path
