from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.pipeline import PipelineFactory, PipelineConfig, PipelineOptions
from medical_kg_nlp.retrieval.adapters import FTSRetrieverAdapter
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.terminology import (
    SQLiteTerminologyRepository,
    build_terminology_index,
    terminology_cache_path,
)


def test_sqlite_exact_and_toneless_lookup_match_in_memory_store(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "concepts.jsonl")
    store = DictionaryStore.from_jsonl(source)
    manifest = build_terminology_index((source,), cache_dir=tmp_path / "cache")
    repository = SQLiteTerminologyRepository(
        manifest.index_path,
        expected_source_paths=(source,),
    )

    expected = store.exact_lookup("Đái tháo đường")
    actual = repository.exact_lookup("Đái tháo đường")

    assert [entry.concept_id for entry in actual] == [entry.concept_id for entry in expected]
    assert repository.toneless_lookup("dai thao duong")[0].code == "E11.9"
    assert repository.get_by_code(CodeSystem.RXNORM, "6809").canonical_name == "metformin"
    assert repository.get_by_concept_id("ICD:E11.9") == expected[0]
    assert repository.contains(CodeSystem.ICD10, "E11.9")
    assert repository.contains(CodeSystem.RXNORM, "6809")
    assert not repository.contains(CodeSystem.ICD10, "MISSING")
    with sqlite3.connect(manifest.index_path) as connection:
        plan = connection.execute(
            "EXPLAIN QUERY PLAN SELECT 1 FROM concepts "
            "WHERE code_system = ? AND code = ? LIMIT 1",
            ("ICD-10", "E11.9"),
        ).fetchall()
    assert any("concepts_code_idx" in str(row) for row in plan)


def test_sqlite_filters_type_and_code_system_before_limit(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "concepts.jsonl")
    manifest = build_terminology_index((source,), cache_dir=tmp_path / "cache")
    repository = SQLiteTerminologyRepository(manifest.index_path)

    results = repository.exact_lookup(
        "shared term",
        entity_type=EntityType.DISEASE,
        code_systems=(CodeSystem.ICD10,),
        limit=1,
    )

    assert [(entry.semantic_type, entry.code_system) for entry in results] == [
        (EntityType.DISEASE, CodeSystem.ICD10)
    ]


def test_sqlite_exact_lookup_prioritizes_canonical_ingredient_over_product_field(
    tmp_path: Path,
) -> None:
    source = tmp_path / "rxnorm.jsonl"
    rows = [
        {
            "concept_id": "RX:6809",
            "code": "6809",
            "code_system": "RxNorm",
            "canonical_name": "metformin",
            "semantic_type": "DRUG",
            "rxnorm_tty": "IN",
            "ingredient": "metformin",
            "source": "test",
        },
        {
            "concept_id": "RX:1000",
            "code": "1000",
            "code_system": "RxNorm",
            "canonical_name": "metformin 500 MG Oral Tablet",
            "semantic_type": "DRUG",
            "rxnorm_tty": "SCD",
            "ingredient": "metformin",
            "source": "test",
        },
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = build_terminology_index((source,), cache_dir=tmp_path / "cache")
    repository = SQLiteTerminologyRepository(manifest.index_path)

    assert [entry.code for entry in repository.exact_lookup("metformin")] == [
        "6809",
        "1000",
    ]
    assert repository.search("metformin")[0].code == "6809"


def test_sqlite_fts_search_is_deterministic_across_threads(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "concepts.jsonl")
    manifest = build_terminology_index((source,), cache_dir=tmp_path / "cache")
    repository = SQLiteTerminologyRepository(manifest.index_path)

    def query(_: int) -> list[str]:
        return [
            entry.concept_id
            for entry in repository.search(
                "metfor",
                entity_type=EntityType.DRUG,
                code_systems=(CodeSystem.RXNORM,),
                limit=5,
            )
        ]

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(query, range(32)))

    assert results == [["RX:6809"]] * 32


def test_sqlite_fts_exposes_alias_similarity_instead_of_rank_placeholder(
    tmp_path: Path,
) -> None:
    source = tmp_path / "concepts.jsonl"
    rows = [
        {
            "concept_id": "ICD:N18.9",
            "code": "N18.9",
            "code_system": "ICD-10",
            "canonical_name": "chronic kidney disease",
            "semantic_type": "DISEASE",
            "source": "fixture",
        },
        {
            "concept_id": "ICD:K76.9",
            "code": "K76.9",
            "code_system": "ICD-10",
            "canonical_name": "chronic liver disease",
            "semantic_type": "DISEASE",
            "source": "fixture",
        },
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = build_terminology_index((source,), cache_dir=tmp_path / "cache")
    repository = SQLiteTerminologyRepository(manifest.index_path)

    hits = repository.search_scored(
        "chronic kidny disease",
        entity_type=EntityType.DISEASE,
        code_systems=(CodeSystem.ICD10,),
        limit=5,
    )
    candidates = FTSRetrieverAdapter(repository).retrieve(
        "chronic kidny disease",
        EntityType.DISEASE,
        "",
        5,
    )

    assert hits[0].entry.code == "N18.9"
    assert hits[0].score > 0.90
    assert hits[0].matched_alias == "chronic kidney disease"
    assert hits[0].match_kind == "partial_tokens"
    assert candidates[0].score == hits[0].score
    assert candidates[0].matched_alias == hits[0].matched_alias


def test_sqlite_fts_caps_lexical_score_for_conflicting_numbers(tmp_path: Path) -> None:
    source = tmp_path / "concepts.jsonl"
    source.write_text(
        json.dumps(
            {
                "concept_id": "RX:200",
                "code": "200",
                "code_system": "RxNorm",
                "canonical_name": "test drug 200 mg oral tablet",
                "semantic_type": "DRUG",
                "source": "fixture",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = build_terminology_index((source,), cache_dir=tmp_path / "cache")
    repository = SQLiteTerminologyRepository(manifest.index_path)

    hits = repository.search_scored(
        "test drug 100 mg oral tablet",
        entity_type=EntityType.DRUG,
        code_systems=(CodeSystem.RXNORM,),
        limit=5,
    )

    assert hits
    assert hits[0].score <= 0.49


def test_sqlite_fts_search_falls_back_to_order_independent_terms(tmp_path: Path) -> None:
    source = tmp_path / "concepts.jsonl"
    source.write_text(
        json.dumps(
            {
                "concept_id": "ICD:C34.9",
                "code": "C34.9",
                "code_system": "ICD-10",
                "canonical_name": "malignant cancer of the lung",
                "semantic_type": "DISEASE",
                "source": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = build_terminology_index((source,), cache_dir=tmp_path / "cache")
    repository = SQLiteTerminologyRepository(manifest.index_path)

    results = repository.search(
        "lung cancer",
        entity_type=EntityType.DISEASE,
        code_systems=(CodeSystem.ICD10,),
        limit=5,
    )

    assert [entry.code for entry in results] == ["C34.9"]


def test_sqlite_fts_search_fills_top_k_with_partial_token_matches(tmp_path: Path) -> None:
    source = tmp_path / "concepts.jsonl"
    rows = [
        {
            "concept_id": "ICD:N18.9",
            "code": "N18.9",
            "code_system": "ICD-10",
            "canonical_name": "chronic kidney disease",
            "semantic_type": "DISEASE",
            "source": "test",
        },
        {
            "concept_id": "ICD:K76.9",
            "code": "K76.9",
            "code_system": "ICD-10",
            "canonical_name": "acute liver disease",
            "semantic_type": "DISEASE",
            "source": "test",
        },
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = build_terminology_index((source,), cache_dir=tmp_path / "cache")
    repository = SQLiteTerminologyRepository(manifest.index_path)

    results = repository.search(
        "advanced chronic disease",
        entity_type=EntityType.DISEASE,
        code_systems=(CodeSystem.ICD10,),
        limit=5,
    )

    assert [entry.code for entry in results] == ["N18.9", "K76.9"]


def test_sqlite_fts_search_does_not_drop_short_numeric_subtype(tmp_path: Path) -> None:
    source = tmp_path / "concepts.jsonl"
    source.write_text(
        json.dumps(
            {
                "concept_id": "ICD:E10",
                "code": "E10",
                "code_system": "ICD-10",
                "canonical_name": "type 1 diabetes mellitus",
                "semantic_type": "DISEASE",
                "source": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = build_terminology_index((source,), cache_dir=tmp_path / "cache")
    repository = SQLiteTerminologyRepository(manifest.index_path)

    results = repository.search(
        "type 2 diabetes",
        entity_type=EntityType.DISEASE,
        code_systems=(CodeSystem.ICD10,),
        limit=5,
    )

    assert results == []


def test_sqlite_repository_rejects_stale_source_fingerprint(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "concepts.jsonl")
    manifest = build_terminology_index((source,), cache_dir=tmp_path / "cache")
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint is stale"):
        SQLiteTerminologyRepository(
            manifest.index_path,
            expected_source_paths=(source,),
        )


def test_sqlite_repository_is_query_only_and_cache_key_is_versioned(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "concepts.jsonl")
    manifest = build_terminology_index((source,), cache_dir=tmp_path / "cache")
    repository = SQLiteTerminologyRepository(manifest.index_path)

    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        repository._connection().execute("DELETE FROM concepts")

    current = terminology_cache_path(tmp_path, (source,), normalization_version="lookup-v1")
    changed = terminology_cache_path(tmp_path, (source,), normalization_version="lookup-v2")
    assert current != changed


def test_sqlite_index_applies_and_fingerprints_strict_alias_overlays(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "concepts.jsonl")
    overlay = _write_overlay(
        tmp_path / "aliases.jsonl",
        target_concept_id="RX:6809",
        alias="metformin immediate release",
    )

    manifest = build_terminology_index(
        (source,),
        alias_overlay_paths=(overlay,),
        cache_dir=tmp_path / "cache",
    )
    repository = SQLiteTerminologyRepository(
        manifest.index_path,
        expected_source_paths=(source,),
        expected_alias_overlay_paths=(overlay,),
    )

    assert repository.exact_lookup("metformin immediate release")[0].code == "6809"
    assert manifest.overlay_alias_count == 1
    assert repository.metadata["alias_overlay_fingerprint"] == (
        manifest.alias_overlay_fingerprint
    )
    changed_cache = terminology_cache_path(
        tmp_path,
        (source,),
        alias_overlay_paths=(overlay,),
    )
    assert changed_cache != terminology_cache_path(tmp_path, (source,))

    overlay.write_text(
        overlay.read_text(encoding="utf-8").replace("immediate", "extended"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="alias-overlay fingerprint is stale"):
        SQLiteTerminologyRepository(
            manifest.index_path,
            expected_source_paths=(source,),
            expected_alias_overlay_paths=(overlay,),
        )


def test_sqlite_index_rejects_invalid_or_conflicting_alias_overlays(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "concepts.jsonl")
    unknown = _write_overlay(
        tmp_path / "unknown.jsonl",
        target_concept_id="RX:missing",
        alias="new alias",
    )
    with pytest.raises(ValueError, match="unknown target concept"):
        build_terminology_index((source,), alias_overlay_paths=(unknown,))

    wrong_type = _write_overlay(
        tmp_path / "wrong-type.jsonl",
        target_concept_id="RX:6809",
        alias="new metformin alias",
        semantic_type="DISEASE",
    )
    with pytest.raises(ValueError, match="semantic_type"):
        build_terminology_index((source,), alias_overlay_paths=(wrong_type,))

    cross_space_homonym = _write_overlay(
        tmp_path / "cross-space.jsonl",
        target_concept_id="RX:6809",
        alias="Đái tháo đường",
    )
    cross_space_manifest = build_terminology_index(
        (source,),
        alias_overlay_paths=(cross_space_homonym,),
        cache_dir=tmp_path / "cross-space-cache",
    )
    cross_space_repository = SQLiteTerminologyRepository(cross_space_manifest.index_path)
    assert cross_space_repository.exact_lookup(
        "Đái tháo đường",
        entity_type=EntityType.DRUG,
        code_systems=(CodeSystem.RXNORM,),
    )[0].code == "6809"

    collision_source = tmp_path / "collision-concepts.jsonl"
    collision_source.write_text(
        json.dumps(
            {
                "concept_id": "RX:999",
                "code": "999",
                "code_system": "RxNorm",
                "canonical_name": "shared drug alias",
                "semantic_type": "DRUG",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    collision = _write_overlay(
        tmp_path / "collision.jsonl",
        target_concept_id="RX:6809",
        alias="shared drug alias",
    )
    with pytest.raises(ValueError, match="already belongs to canonical concepts"):
        build_terminology_index(
            (source, collision_source),
            alias_overlay_paths=(collision,),
        )


def test_sqlite_index_merges_compatible_duplicate_concepts(tmp_path: Path) -> None:
    first = _write_source(tmp_path / "first.jsonl")
    second = tmp_path / "second.jsonl"
    second.write_text(
        json.dumps(
            {
                "concept_id": "RX:6809",
                "code": "6809",
                "code_system": "RxNorm",
                "canonical_name": "metformin hydrochloride",
                "semantic_type": "DRUG",
                "aliases": ["metformin HCl"],
                "rxnorm_tty": "IN",
                "ingredient": "metformin",
                "source": "second",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = build_terminology_index((first, second), cache_dir=tmp_path / "cache")
    repository = SQLiteTerminologyRepository(manifest.index_path)

    assert manifest.concept_count == 2
    assert repository.exact_lookup("metformin HCl")[0].concept_id == "RX:6809"
    assert repository.exact_lookup("metformin hydrochloride")[0].concept_id == "RX:6809"


def test_pipeline_factory_requires_and_uses_prebuilt_normalization_index(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "concepts.jsonl")
    missing_index = tmp_path / "missing.sqlite3"
    options = PipelineOptions(
        candidate_sources=("exact",),
        enable_context=False,
        enable_entity_kg_validation=False,
        enable_relations=False,
        enable_relation_kg_validation=False,
    )
    config = PipelineConfig(
        recognition_dictionary_path=str(source),
        normalization_dictionary_paths=(str(source),),
        normalization_index_path=str(missing_index),
        options=options,
    )

    with pytest.raises(FileNotFoundError, match="Build it explicitly first"):
        PipelineFactory.from_config(config)

    manifest = build_terminology_index((source,), output_path=missing_index)
    runner = PipelineFactory.from_config(config)
    repository = runner.components.terminology_repository

    assert manifest.index_path == str(missing_index)
    assert repository is not None
    assert repository.get_by_code(CodeSystem.RXNORM, "6809").concept_id == "RX:6809"


def _write_source(path: Path) -> Path:
    rows = [
        {
            "concept_id": "ICD:E11.9",
            "code": "E11.9",
            "code_system": "ICD-10",
            "canonical_name": "Đái tháo đường",
            "semantic_type": "DISEASE",
            "aliases": ["shared term"],
            "source": "test",
        },
        {
            "concept_id": "RX:6809",
            "code": "6809",
            "code_system": "RxNorm",
            "canonical_name": "metformin",
            "semantic_type": "DRUG",
            "aliases": ["shared term", "metformin hydrochloride"],
            "rxnorm_tty": "IN",
            "ingredient": "metformin",
            "source": "test",
        },
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _write_overlay(
    path: Path,
    *,
    target_concept_id: str,
    alias: str,
    semantic_type: str = "DRUG",
) -> Path:
    path.write_text(
        json.dumps(
            {
                "alias": alias,
                "target_concept_id": target_concept_id,
                "semantic_type": semantic_type,
                "code_system": "RxNorm",
                "code": "6809",
                "source": "test-mining",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path
