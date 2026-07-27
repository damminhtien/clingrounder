"""Corpus-derived recognition knowledge promotion tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import yaml

from medical_kg_nlp.cli.main import main
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.mining.io import write_jsonl
from medical_kg_nlp.mining.lexicon import MentionInventoryEntry
from medical_kg_nlp.mining.recognition_knowledge import (
    RecognitionKnowledgePolicy,
    compile_recognition_knowledge,
)
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.utils.hashing import sha256_file

_INVENTORY_SHA256 = "a" * 64


def _entry(
    term_id: str,
    mention: str,
    *,
    source_label: str = "Symptom_and_Disease",
    entity_type: str = "FINDING",
    document_count: int = 2,
    consensus_count: int = 0,
) -> MentionInventoryEntry:
    return MentionInventoryEntry(
        term_id=term_id,
        normalized_mention=mention.casefold(),
        entity_type=entity_type,
        source_label=source_label,
        occurrence_count=max(2, document_count),
        document_count=document_count,
        consensus_occurrence_count=consensus_count,
        surface_variant_count=1,
        surface_variants=((mention, max(2, document_count)),),
        source_artifact_ids=("artifact:fixture",),
        label_sources=(("source_human_annotation", max(2, document_count)),),
        example_document_ids=("doc-1", "doc-2")[:document_count],
        review_tier=(
            "duplicate_consensus_supported" if consensus_count else "multi_document"
        ),
        recommended_use="terminology_alias_review",
    )


def _policy(*, inventory_sha256: str = _INVENTORY_SHA256) -> RecognitionKnowledgePolicy:
    return RecognitionKnowledgePolicy(
        policy_id="vietbioner-recognition-v1",
        accepted_inventory_sha256=(inventory_sha256,),
        source_label_types=(
            ("DiagnosticProcedure", EntityType.PROCEDURE),
            ("Symptom_and_Disease", EntityType.FINDING),
        ),
        accepted_label_sources=(
            "source_human_annotation",
            "exact_duplicate_consensus",
        ),
        accepted_review_tiers=(
            "multi_document",
            "duplicate_consensus_supported",
        ),
    )


def test_compiler_promotes_supported_mentions_without_medical_codes() -> None:
    result = compile_recognition_knowledge(
        (_entry("term:1", "Lao phổi"),),
        _policy(),
        inventory_sha256=_INVENTORY_SHA256,
    )

    assert len(result.concepts) == 1
    concept = result.concepts[0]
    assert concept["canonical_name"] == "Lao phổi"
    assert concept["semantic_type"] == "FINDING"
    assert concept["code_system"] == "NONE"
    assert concept["code"] is None
    assert result.report["promotion_contract"].endswith("code-free")


def test_compiler_rejects_mapped_and_baseline_type_conflicts() -> None:
    same_surface_conflict = (
        _entry("term:finding", "PCR"),
        _entry(
            "term:procedure",
            "PCR",
            source_label="DiagnosticProcedure",
            entity_type="PROCEDURE",
        ),
    )
    conflict_result = compile_recognition_knowledge(
        same_surface_conflict,
        _policy(),
        inventory_sha256=_INVENTORY_SHA256,
    )
    assert not conflict_result.concepts
    assert conflict_result.report["reason_counts"] == {"mapped_type_conflict": 2}

    baseline = ConceptEntry(
        concept_id="ICD:A15.0",
        code="A15.0",
        code_system=CodeSystem.ICD10,
        canonical_name="lao phổi",
        semantic_type=EntityType.DISEASE,
    )
    baseline_result = compile_recognition_knowledge(
        (_entry("term:baseline", "Lao phổi"),),
        _policy(),
        inventory_sha256=_INVENTORY_SHA256,
        baseline_entries=(baseline,),
    )
    assert not baseline_result.concepts
    assert baseline_result.report["reason_counts"] == {"baseline_type_conflict": 1}


def test_compiler_can_add_reviewed_code_free_type_evidence_for_baseline_alias() -> None:
    baseline = ConceptEntry(
        concept_id="ICD:K59.0",
        code="K59.0",
        code_system=CodeSystem.ICD10,
        canonical_name="táo bón",
        semantic_type=EntityType.DISEASE,
    )
    policy = replace(
        _policy(),
        allow_reviewed_baseline_type_conflicts=True,
        accepted_source_mentions=frozenset(
            {("Symptom_and_Disease", "táo bón")}
        ),
    )

    result = compile_recognition_knowledge(
        (_entry("term:symptom", "Táo bón"),),
        policy,
        inventory_sha256=_INVENTORY_SHA256,
        baseline_entries=(baseline,),
    )

    assert len(result.concepts) == 1
    concept = result.concepts[0]
    assert concept["semantic_type"] == "FINDING"
    assert concept["code_system"] == "NONE"
    assert concept["code"] is None
    assert result.report["reason_counts"] == {
        "reviewed_baseline_type_evidence": 1
    }


def test_compiler_fail_closes_on_source_aware_reviewed_mentions() -> None:
    base_policy = _policy()
    policy = RecognitionKnowledgePolicy(
        **{
            **base_policy.__dict__,
            "accepted_source_mentions": frozenset(
                {("Symptom_and_Disease", "lao phổi")}
            ),
        }
    )

    result = compile_recognition_knowledge(
        (
            _entry("term:accepted", "Lao phổi"),
            _entry("term:unreviewed", "Ho ra máu"),
        ),
        policy,
        inventory_sha256=_INVENTORY_SHA256,
    )

    assert [row["canonical_name"] for row in result.concepts] == ["Lao phổi"]
    assert result.report["reviewed_source_mention_count"] == 1
    assert result.report["reason_counts"]["mention_not_reviewed"] == 1


def test_compiler_blocks_short_alias_even_when_reviewed() -> None:
    base_policy = _policy()
    policy = replace(
        base_policy,
        min_alias_characters=3,
        accepted_source_mentions=frozenset({("Symptom_and_Disease", "đỏ")}),
    )

    result = compile_recognition_knowledge(
        (_entry("term:short", "đỏ"),),
        policy,
        inventory_sha256=_INVENTORY_SHA256,
    )

    assert not result.concepts
    assert result.report["reason_counts"] == {"alias_shape_not_allowed": 1}


def test_compile_recognition_cli_pins_inventory_and_writes_audit_artifacts(
    tmp_path: Path,
    capsys,
) -> None:
    inventory_path = tmp_path / "inventory.jsonl"
    write_jsonl(inventory_path, (_entry("term:1", "PCR", source_label="DiagnosticProcedure", entity_type="PROCEDURE").to_dict(),))
    inventory_sha256 = sha256_file(inventory_path)
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "mined-recognition-promotion-policy.v1",
                "policy_id": "fixture-recognition-v1",
                "accepted_inventory_sha256": [inventory_sha256],
                "source_label_types": {"DiagnosticProcedure": "PROCEDURE"},
                "accepted_label_sources": ["source_human_annotation"],
                "accepted_review_tiers": ["multi_document"],
                "accepted_source_mentions": {"DiagnosticProcedure": ["PCR"]},
                "min_occurrences": 2,
                "min_documents": 2,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "recognition.jsonl"
    report_path = tmp_path / "report.json"

    exit_code = main(
        [
            "data",
            "knowledge",
            "compile-recognition",
            "--inventory",
            str(inventory_path),
            "--policy",
            str(policy_path),
            "--output",
            str(output_path),
            "--decisions-output",
            str(tmp_path / "decisions.jsonl"),
            "--report-output",
            str(report_path),
        ]
    )
    capsys.readouterr()
    concept = json.loads(output_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert concept["semantic_type"] == "PROCEDURE"
    assert report["inventory_sha256"] == inventory_sha256
    assert report["reviewed_source_mention_count"] == 1
    assert report["outputs"]["recognition_dictionary_sha256"] == sha256_file(output_path)
