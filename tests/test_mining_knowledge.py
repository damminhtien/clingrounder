"""Mined aliases must pass provenance, terminology, and ambiguity gates."""

from __future__ import annotations

import json
from pathlib import Path

from medical_kg_nlp.mining.io import write_jsonl
from medical_kg_nlp.mining.knowledge import (
    MinedAliasPromotionPolicy,
    compile_mined_aliases,
    load_alias_promotion_policy,
)
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.terminology import SQLiteTerminologyRepository, build_terminology_index

_SOURCE_SHA = "a" * 64


def test_compiler_promotes_only_unique_source_pinned_aliases(tmp_path: Path) -> None:
    source = _write_terminology(tmp_path / "rxnorm.jsonl")
    base_manifest = build_terminology_index((source,), cache_dir=tmp_path / "base-cache")
    repository = SQLiteTerminologyRepository(base_manifest.index_path)
    proposals = (
        _proposal("p1", "100", "drug one hundred", "Drug One Hundred", "SY"),
        _proposal("p1b", "100", "drug one hundred", "DRUG ONE HUNDRED", "SCD"),
        _proposal("p2", "100", "shared collision", "Shared Collision", "SY"),
        _proposal("p3", "200", "shared collision", "shared collision", "SBD"),
        _proposal("p4", "999", "unknown drug", "Unknown Drug", "SY"),
        _proposal("p5", "100", "package only", "Package Only", "BPCK"),
        _proposal("p6", "100", "brand 200", "Brand 200", "SY"),
        _proposal("p7", "100", "drug 100", "Drug 100", "SCD"),
        {
            **_proposal("p8", "100", "wrong source", "Wrong Source", "SY"),
            "source": "untrusted",
        },
    )

    result = compile_mined_aliases(proposals, repository, _policy())
    repository.close()

    assert [(row["code"], row["normalized_alias"]) for row in result.alias_overlays] == [
        ("100", "drug one hundred")
    ]
    assert result.alias_overlays[0]["proposal_ids"] == ["p1", "p1b"]
    assert result.recognition_concepts[0]["aliases"] == ["Drug One Hundred"]
    assert result.report["decision_counts"] == {
        "promoted": 2,
        "rejected": 6,
        "skipped": 1,
    }
    assert result.report["reason_counts"] == {
        "already_present": 1,
        "canonical_alias_conflict": 1,
        "proposal_target_conflict": 2,
        "source_not_allowed": 1,
        "tty_not_allowed": 1,
        "unique_reviewed_alias": 2,
        "unknown_terminology_code": 1,
    }

    overlay = tmp_path / "compiled-aliases.jsonl"
    write_jsonl(overlay, result.alias_overlays)
    enriched = build_terminology_index(
        (source,),
        alias_overlay_paths=(overlay,),
        cache_dir=tmp_path / "enriched-cache",
    )
    enriched_repository = SQLiteTerminologyRepository(
        enriched.index_path,
        expected_source_paths=(source,),
        expected_alias_overlay_paths=(overlay,),
    )
    assert enriched_repository.exact_lookup("drug one hundred")[0].code == "100"


def test_policy_loader_requires_pinned_source_and_typed_targets(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        """\
schema_version: mined-alias-promotion-policy.v1
policy_id: official-drugs-v1
accepted_sources: [official]
accepted_source_sha256:
  - aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
accepted_review_statuses: [review_required]
allowed_code_systems: [RxNorm]
allowed_semantic_types: [DRUG]
allowed_ttys: [SCD, SBD, SY]
min_supporting_records: 2
min_alias_characters: 4
max_alias_characters: 200
max_alias_tokens: 30
allow_numeric_only: false
""",
        encoding="utf-8",
    )

    policy = load_alias_promotion_policy(policy_path)

    assert policy.policy_id == "official-drugs-v1"
    assert policy.allowed_code_systems == (CodeSystem.RXNORM,)
    assert policy.allowed_semantic_types == (EntityType.DRUG,)
    assert policy.min_supporting_records == 2


def _policy() -> MinedAliasPromotionPolicy:
    return MinedAliasPromotionPolicy(
        policy_id="test-official-rxnorm-v1",
        accepted_sources=("official",),
        accepted_source_sha256=(_SOURCE_SHA,),
        accepted_review_statuses=("review_required",),
        allowed_code_systems=(CodeSystem.RXNORM,),
        allowed_semantic_types=(EntityType.DRUG,),
        allowed_ttys=("SCD", "SBD", "PSN", "SY"),
        min_supporting_records=1,
        min_alias_characters=3,
        max_alias_characters=200,
        max_alias_tokens=30,
    )


def _proposal(
    proposal_id: str,
    code: str,
    normalized_alias: str,
    surface: str,
    tty: str,
) -> dict[str, object]:
    return {
        "proposal_id": proposal_id,
        "code_system": "RxNorm",
        "code": code,
        "normalized_alias": normalized_alias,
        "surface_variants": [{"surface": surface, "ttys": [tty]}],
        "supporting_set_version_count": 2,
        "source": "official",
        "source_version": "2026-07-17",
        "source_sha256": _SOURCE_SHA,
        "review_status": "review_required",
    }


def _write_terminology(path: Path) -> Path:
    rows = [
        {
            "concept_id": "RX:100",
            "code": "100",
            "code_system": "RxNorm",
            "canonical_name": "Drug 100",
            "semantic_type": "DRUG",
            "rxnorm_tty": "SCD",
            "source": "rxnorm-test",
        },
        {
            "concept_id": "RX:200",
            "code": "200",
            "code_system": "RxNorm",
            "canonical_name": "Brand 200",
            "semantic_type": "DRUG",
            "rxnorm_tty": "SBD",
            "source": "rxnorm-test",
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path
