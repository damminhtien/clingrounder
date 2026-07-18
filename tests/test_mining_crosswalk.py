"""Exact terminology crosswalk tests for mined mention inventories."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from medical_kg_nlp.cli.main import main
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.mining.crosswalk import MentionCrosswalkPolicy, crosswalk_mentions
from medical_kg_nlp.mining.io import write_jsonl
from medical_kg_nlp.mining.lexicon import MentionInventoryEntry, load_mention_inventory
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.terminology import (
    InMemoryTerminologyRepository,
    SQLiteTerminologyRepository,
    build_terminology_index,
)


def _inventory(
    term_id: str,
    mention: str,
    *,
    entity_type: str = "FINDING",
    source_label: str | None = "Symptom_and_Disease",
    occurrences: int = 1,
) -> MentionInventoryEntry:
    return MentionInventoryEntry(
        term_id=term_id,
        normalized_mention=mention,
        entity_type=entity_type,
        source_label=source_label,
        occurrence_count=occurrences,
        document_count=1,
        consensus_occurrence_count=0,
        surface_variant_count=1,
        surface_variants=((mention, occurrences),),
        source_artifact_ids=("artifact:test",),
        label_sources=(("source_human_annotation", occurrences),),
        example_document_ids=("doc:test",),
        review_tier="single_occurrence",
        recommended_use="terminology_alias_review",
    )


def _concept(
    concept_id: str,
    code: str | None,
    name: str,
    *,
    alias: str | None = None,
) -> ConceptEntry:
    return ConceptEntry(
        concept_id=concept_id,
        code=code,
        code_system=CodeSystem.ICD10,
        canonical_name=name,
        semantic_type=EntityType.DISEASE,
        aliases=() if alias is None else (alias,),
        source="fixture",
    )


def _policy() -> MentionCrosswalkPolicy:
    return MentionCrosswalkPolicy(
        policy_id="finding-to-icd-v1",
        source_entity_type="FINDING",
        source_label="Symptom_and_Disease",
        target_entity_types=(EntityType.DISEASE,),
        code_systems=(CodeSystem.ICD10,),
    )


def test_crosswalk_separates_unique_ambiguous_unmatched_and_skipped() -> None:
    repository = InMemoryTerminologyRepository(
        DictionaryStore(
            [
                _concept("ICD:R05", "R05", "ho"),
                _concept("ICD:E11:a", "E11", "đái tháo đường"),
                _concept("ICD:E11:b", "E11", "bệnh đái tháo đường", alias="đái tháo đường"),
                _concept("ICD:R52", "R52", "đau"),
                _concept("ICD:R07", "R07", "đau ngực", alias="đau"),
                _concept("LOCAL:no-code", None, "khái niệm chưa mã"),
            ]
        )
    )
    entries = (
        _inventory("term:unique", "ho", occurrences=3),
        _inventory("term:same-code", "đái tháo đường"),
        _inventory("term:ambiguous", "đau"),
        _inventory("term:no-code", "khái niệm chưa mã"),
        _inventory("term:missing", "không có trong từ điển"),
        _inventory("term:skipped", "ho", entity_type="OTHER", source_label=None),
    )

    result = crosswalk_mentions(entries, repository, (_policy(),), workers=2)
    statuses = {record.term_id: record.status for record in result.records}

    assert statuses == {
        "term:ambiguous": "ambiguous_code_exact",
        "term:missing": "unmatched",
        "term:no-code": "concept_only_exact",
        "term:same-code": "unique_code_exact",
        "term:skipped": "skipped_no_policy",
        "term:unique": "unique_concept_exact",
    }
    unique = next(record for record in result.records if record.term_id == "term:unique")
    assert unique.to_dict()["promotion_status"] == "review_required"
    assert result.report["unique_exact_entry_count"] == 2
    assert result.report["unique_exact_occurrence_count"] == 4


def test_crosswalk_is_deterministic_across_worker_counts() -> None:
    repository = InMemoryTerminologyRepository(DictionaryStore([_concept("ICD:R05", "R05", "ho")]))
    entries = tuple(_inventory(f"term:{index}", "ho") for index in range(12))

    serial = crosswalk_mentions(entries, repository, (_policy(),), workers=1)
    concurrent = crosswalk_mentions(entries, repository, (_policy(),), workers=4)

    assert [record.to_dict() for record in serial.records] == [
        record.to_dict() for record in concurrent.records
    ]


def test_crosswalk_lexical_fallback_is_opt_in_and_review_only(tmp_path: Path) -> None:
    source = tmp_path / "terminology.jsonl"
    source.write_text(
        json.dumps(
            {
                "concept_id": "ICD:C34.9",
                "code": "C34.9",
                "code_system": "ICD-10",
                "canonical_name": "malignant cancer of the lung",
                "semantic_type": "DISEASE",
                "source": "fixture",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    index = tmp_path / "terminology.sqlite3"
    build_terminology_index((source,), output_path=index)
    repository = SQLiteTerminologyRepository(index)
    entries = (_inventory("term:lung-cancer", "lung cancer"),)

    exact = crosswalk_mentions(entries, repository, (_policy(),))
    lexical = crosswalk_mentions(
        entries,
        repository,
        (_policy(),),
        lexical_fallback=True,
    )

    assert exact.records[0].status == "unmatched"
    row = lexical.records[0].to_dict()
    assert row["match_mode"] == "fts_lexical"
    assert row["status"] == "lexical_candidates"
    assert row["promotion_status"] == "review_required"
    assert row["automatic_promotion_allowed"] is False
    assert row["candidates"][0]["code"] == "C34.9"
    assert lexical.report["unique_exact_entry_count"] == 0
    repository.close()


def test_crosswalk_policy_rejects_type_code_system_mismatch() -> None:
    with pytest.raises(ValueError, match="cannot map DRUG to ICD-10"):
        MentionCrosswalkPolicy(
            policy_id="invalid",
            source_entity_type="FINDING",
            source_label=None,
            target_entity_types=(EntityType.DRUG,),
            code_systems=(CodeSystem.ICD10,),
        )


def test_crosswalk_cli_validates_index_and_writes_deterministic_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "terminology.jsonl"
    source.write_text(
        json.dumps(
            {
                "concept_id": "ICD:R05",
                "code": "R05",
                "code_system": "ICD-10",
                "canonical_name": "ho",
                "semantic_type": "DISEASE",
                "source": "fixture",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    index = tmp_path / "terminology.sqlite3"
    overlay = tmp_path / "aliases.jsonl"
    overlay.write_text(
        json.dumps(
            {
                "alias": "cough",
                "target_concept_id": "ICD:R05",
                "semantic_type": "DISEASE",
                "code_system": "ICD-10",
                "code": "R05",
                "source": "fixture-overlay",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    build_terminology_index(
        (source,),
        alias_overlay_paths=(overlay,),
        output_path=index,
    )
    inventory_path = tmp_path / "inventory.jsonl"
    write_jsonl(inventory_path, (_inventory("term:ho", "ho").to_dict(),))
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "medical-mention-crosswalk-policy.v1",
                "policies": [
                    {
                        "policy_id": "finding-to-icd-v1",
                        "source_entity_type": "FINDING",
                        "source_label": "Symptom_and_Disease",
                        "target_entity_types": ["DISEASE"],
                        "code_systems": ["ICD-10"],
                    }
                ],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "crosswalk.jsonl"
    report = tmp_path / "report.json"

    exit_code = main(
        [
            "data",
            "lexicon",
            "crosswalk",
            "--inventory",
            str(inventory_path),
            "--index",
            str(index),
            "--source",
            str(source),
            "--alias-overlay-source",
            str(overlay),
            "--policy",
            str(policy_path),
            "--output",
            str(output),
            "--report-output",
            str(report),
            "--workers",
            "2",
        ]
    )
    stdout = json.loads(capsys.readouterr().out)
    row = json.loads(output.read_text(encoding="utf-8"))
    report_payload = json.loads(report.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert stdout["unique_exact_entry_count"] == 1
    assert row["status"] == "unique_concept_exact"
    assert row["candidates"][0]["code"] == "R05"
    assert report_payload["terminology"]["source_fingerprint"]
    assert load_mention_inventory(inventory_path)[0].term_id == "term:ho"

    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint is stale"):
        main(
            [
                "data",
                "lexicon",
                "crosswalk",
                "--inventory",
                str(inventory_path),
                "--index",
                str(index),
                "--source",
                str(source),
                "--alias-overlay-source",
                str(overlay),
                "--policy",
                str(policy_path),
                "--output",
                str(output),
                "--report-output",
                str(report),
            ]
        )
