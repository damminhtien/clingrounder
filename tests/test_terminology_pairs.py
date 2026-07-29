from __future__ import annotations

import hashlib
import json
from pathlib import Path

from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.training.terminology_pairs import (
    SynonymPairMode,
    TerminologyPairConfig,
    build_terminology_synonym_pairs,
    write_terminology_pair_dataset,
)


def test_synonym_pairs_are_same_concept_typed_and_deterministic() -> None:
    entries = [
        ConceptEntry(
            concept_id="ICD:I10",
            code="I10",
            code_system=CodeSystem.ICD10,
            canonical_name="Tăng huyết áp",
            official_name_en="Hypertension",
            semantic_type=EntityType.DISEASE,
            aliases=("cao huyết áp", "tăng  huyết áp"),
            abbreviations=("THA",),
            blocked_aliases=("THA",),
            source="TT06",
        ),
        ConceptEntry(
            concept_id="RX:860975",
            code="860975",
            code_system=CodeSystem.RXNORM,
            canonical_name="metformin",
            semantic_type=EntityType.DRUG,
            aliases=("metformin hydrochloride",),
            source="RxNorm",
        ),
    ]
    config = TerminologyPairConfig(mode=SynonymPairMode.ALL_PAIRS)

    first = build_terminology_synonym_pairs(entries, config=config)
    second = build_terminology_synonym_pairs(reversed(entries), config=config)

    assert first == second
    assert {pair.concept_id for pair in first} == {"ICD:I10", "RX:860975"}
    assert all(pair.left != "THA" and pair.right != "THA" for pair in first)
    assert all(
        pair.entity_type == EntityType.DISEASE
        for pair in first
        if pair.concept_id == "ICD:I10"
    )
    assert all(
        pair.code_system == CodeSystem.RXNORM
        for pair in first
        if pair.concept_id == "RX:860975"
    )


def test_pair_generation_caps_quadratic_alias_expansion() -> None:
    entry = ConceptEntry(
        concept_id="C1",
        code="C1",
        code_system=CodeSystem.LOCAL,
        canonical_name="canonical",
        semantic_type=EntityType.SYMPTOM,
        aliases=tuple(f"alias {index}" for index in range(100)),
    )
    config = TerminologyPairConfig(
        mode=SynonymPairMode.ALL_PAIRS,
        max_names_per_concept=8,
        max_pairs_per_concept=5,
    )

    pairs = build_terminology_synonym_pairs([entry], config=config)

    assert len(pairs) == 5


def test_pair_dataset_writes_fingerprinted_manifest(tmp_path: Path) -> None:
    entry = ConceptEntry(
        concept_id="C1",
        code="C1",
        code_system=CodeSystem.LOCAL,
        canonical_name="đau ngực",
        semantic_type=EntityType.SYMPTOM,
        aliases=("chest pain",),
        source="test",
    )
    config = TerminologyPairConfig()
    pairs = build_terminology_synonym_pairs([entry], config=config)
    output = tmp_path / "pairs.jsonl"

    manifest = write_terminology_pair_dataset(
        pairs,
        output,
        config=config,
        source_fingerprints={"dictionary.jsonl": "a" * 64},
    )

    raw = output.read_bytes()
    row = json.loads(raw.decode("utf-8"))
    assert row["label"] == 1
    assert row["entity_type"] == "SYMPTOM"
    assert manifest["dataset_sha256"] == hashlib.sha256(raw).hexdigest()
    assert output.with_suffix(".jsonl.manifest.json").exists()
