from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.pipeline import PipelineFactory, PipelineFactoryConfig, PipelineOptions
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
    config = PipelineFactoryConfig(
        recognition_dictionary_path=str(source),
        normalization_dictionary_path=str(source),
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
