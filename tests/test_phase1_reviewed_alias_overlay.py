from __future__ import annotations

import json
from pathlib import Path

import pytest

from medical_kg_nlp.benchmarks.phase1.reviewed_alias_overlay import (
    compile_reviewed_candidate_aliases,
    load_reviewed_candidate_proposals,
)
from medical_kg_nlp.terminology import SQLiteTerminologyRepository, build_terminology_index


def test_reviewed_map_compiles_to_typed_normalization_overlay(tmp_path: Path) -> None:
    source = tmp_path / "terminology.jsonl"
    source.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in (
                _concept("ICD10:I10", "I10", "ICD-10", "DISEASE", "tăng huyết áp"),
                _concept("RX:6809", "6809", "RxNorm", "DRUG", "metformin"),
            )
        ),
        encoding="utf-8",
    )
    map_path = tmp_path / "reviewed.jsonl"
    map_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in (
                _map_row("CHẨN_ĐOÁN", "tăng huyết áp nguyên phát", "I10", "candidate_icd"),
                _map_row("THUỐC", "metformin xr", "6809", "candidate_rxnorm_ingredient"),
            )
        ),
        encoding="utf-8",
    )
    manifest = build_terminology_index((source,), cache_dir=tmp_path / "cache")
    repository = SQLiteTerminologyRepository(manifest.index_path, expected_source_paths=(source,))
    result, source_sha = compile_reviewed_candidate_aliases(map_path, repository)
    repository.close()

    assert len(source_sha) == 64
    assert [(row["semantic_type"], row["code"]) for row in result.alias_overlays] == [
        ("DISEASE", "I10"),
        ("DRUG", "6809"),
    ]
    assert all("document_id" not in row and "position" not in row for row in result.alias_overlays)


def test_reviewed_map_rejects_cross_system_and_conflicting_rows(tmp_path: Path) -> None:
    path = tmp_path / "reviewed.jsonl"
    path.write_text(
        json.dumps(
            _map_row(
                "CHẨN_ĐOÁN",
                "tăng huyết áp",
                "6809",
                "candidate_icd",
                code_system="RxNorm",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires ICD-10"):
        load_reviewed_candidate_proposals(path)

    path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                _map_row("CHẨN_ĐOÁN", "tăng huyết áp", "I10", "candidate_icd"),
                _map_row("CHẨN_ĐOÁN", "tăng huyết áp", "I11", "candidate_icd"),
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conflicting codes"):
        load_reviewed_candidate_proposals(path)


def _map_row(
    entity_type: str,
    mention: str,
    code: str,
    stage: str,
    *,
    code_system: str | None = None,
) -> dict[str, object]:
    return {
        "normalized_mention": mention,
        "entity_type": entity_type,
        "candidate": code,
        "code_system": code_system or ("ICD-10" if entity_type == "CHẨN_ĐOÁN" else "RxNorm"),
        "candidate_stage": stage,
        "confidence_tier": "high",
        "occurrence_support": 1,
        "document_support": 1,
        "dictionary_release": "test-release",
        "provenance": "manual_gold_train",
        "rule_id": f"rule-{entity_type}-{mention}-{code}",
        "review_status": "reviewed",
    }


def _concept(
    concept_id: str,
    code: str,
    code_system: str,
    semantic_type: str,
    canonical_name: str,
) -> dict[str, str]:
    return {
        "concept_id": concept_id,
        "code": code,
        "code_system": code_system,
        "semantic_type": semantic_type,
        "canonical_name": canonical_name,
        "source": "test",
    }
