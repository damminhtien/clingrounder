"""Explicit abbreviation mining must preserve offsets and frozen-split isolation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from medical_kg_nlp.cli.parser import build_parser
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.mining.abbreviations import (
    AbbreviationMiningPolicy,
    benchmark_abbreviation_knowledge,
    build_runtime_abbreviation_table,
    load_abbreviation_mining_policy,
    mine_abbreviations,
)
from medical_kg_nlp.mining.records import (
    AccessClass,
    MinedDocument,
    RedistributionPolicy,
    SourceArtifact,
    StoredObject,
)
from medical_kg_nlp.schema.types import CodeSystem, EntityType

_SOURCE_SHA256 = "a" * 64
_SOURCE_VERSION = "fixture-2026-07-22"


def test_miner_aligns_both_definition_directions_and_preserves_raw_offsets() -> None:
    text = (
        "Prefix. Drug Reaction with Eosinophilia and Systemic Symptoms (DRESS). "
        "MRI (magnetic resonance imaging). La resonancia magnética (RM). "
        "Oral (oral route). type II (II)."
    )
    document = _document("doc-1", text, group_id="group-1")

    result = mine_abbreviations(
        (document,),
        (_artifact(),),
        {document.document_id: "train"},
        _policy(minimum_support=1),
    )

    assert [row["abbreviation"] for row in result.definitions] == ["DRESS", "MRI", "RM"]
    assert [row["expansion"] for row in result.definitions] == [
        "Drug Reaction with Eosinophilia and Systemic Symptoms",
        "magnetic resonance imaging",
        "resonancia magnética",
    ]
    assert [row["direction"] for row in result.definitions] == [
        "long_short",
        "short_long",
        "long_short",
    ]
    assert {row["alignment_method"] for row in result.definitions} == {"token_initials"}
    for row in result.definitions:
        abbreviation_start, abbreviation_end = row["abbreviation_span"]
        expansion_start, expansion_end = row["expansion_span"]
        assert text[abbreviation_start:abbreviation_end] == row["abbreviation"]
        assert text[expansion_start:expansion_end] == row["expansion"]
    assert result.report["reason_counts"]["invalid_short_form"] == 2


def test_evaluation_definitions_never_change_runtime_table_or_train_conflicts() -> None:
    training_text = "Drug Reaction with Eosinophilia and Systemic Symptoms (DRESS)."
    documents = (
        _document("train-1", training_text, group_id="group-1"),
        _document("train-2", training_text, group_id="group-2"),
        _document("dev-1", "Drug rash induced severe syndrome (DRESS).", group_id="group-3"),
        _document("dev-2", "Direct renin inhibitor (DRI).", group_id="group-4"),
    )
    splits = {
        "train-1": "train",
        "train-2": "train",
        "dev-1": "development",
        "dev-2": "development",
    }

    result = mine_abbreviations(documents, (_artifact(),), splits, _policy())

    assert len(result.abbreviation_table) == 1
    assert result.abbreviation_table[0]["abbreviation"] == "DRESS"
    assert result.abbreviation_table[0]["expansions"] == [
        "Drug Reaction with Eosinophilia and Systemic Symptoms"
    ]
    assert result.conflicts == ()
    assert result.report["definition_split_counts"] == {"development": 2, "train": 2}
    assert result.report["split_contract"]["evaluation_used_for_table"] is False


def test_initialism_alignment_does_not_truncate_first_acronym_word() -> None:
    text = (
        "Proton Pump Inhibitors (PPIs). "
        "Addiction Research Center Inventories (ARCI). "
        "an adeno-associated virus (AAV). "
        "Beta-Blocker Heart Attack Trial (BHAT). "
        "COPERNICUS) In a double-blind trial (COPERNICUS). "
        "HBeAg-positive subjects (HBeAg)."
    )
    document = _document("doc-initials", text, group_id="group-initials")

    result = mine_abbreviations(
        (document,),
        (_artifact(),),
        {document.document_id: "train"},
        _policy(minimum_support=1),
    )

    assert [(row["abbreviation"], row["expansion"]) for row in result.definitions] == [
        ("PPIs", "Proton Pump Inhibitors"),
        ("ARCI", "Addiction Research Center Inventories"),
        ("AAV", "adeno-associated virus"),
        ("BHAT", "Beta-Blocker Heart Attack Trial"),
    ]
    assert result.report["reason_counts"]["long_form_contains_bracket"] == 1
    assert result.report["reason_counts"]["long_form_starts_with_short_form"] == 1


def test_train_and_baseline_conflicts_are_fail_closed() -> None:
    train_conflict = (
        _document("train-a", "Alpha beta complex (ABC).", group_id="group-a"),
        _document("train-b", "Alternative binding component (ABC).", group_id="group-b"),
    )
    result = mine_abbreviations(
        train_conflict,
        (_artifact(),),
        {document.document_id: "train" for document in train_conflict},
        _policy(minimum_support=1),
    )
    assert result.abbreviation_table == ()
    assert result.conflicts[0]["reason"] == "mined_expansion_conflict"

    baseline_conflict = (
        _document("train-c", "magnetic resonance imaging (MRI).", group_id="group-c"),
    )
    result = mine_abbreviations(
        baseline_conflict,
        (_artifact(),),
        {"train-c": "train"},
        _policy(minimum_support=1),
        base_abbreviation_rows=({"abbreviation": "MRI", "expansions": ["mitral regurgitation"]},),
    )
    assert result.abbreviation_table == ()
    assert result.conflicts[0]["reason"] == "base_table_conflict"


def test_character_alignment_requires_broader_independent_support() -> None:
    text = "hydroxypropyl methylcellulose (HPMC)."
    documents = (
        _document("train-char-1", text, group_id="group-char-1"),
        _document("train-char-2", text, group_id="group-char-2"),
    )

    result = mine_abbreviations(
        documents,
        (_artifact(),),
        {document.document_id: "train" for document in documents},
        _policy(),
    )

    assert result.definitions[0]["alignment_method"] == "backward_characters"
    assert result.abbreviation_table == ()
    assert result.candidates[0]["reason"] == ("insufficient_character_alignment_document_support")


def test_heldout_benchmark_measures_expansion_and_dictionary_retrieval_gain() -> None:
    expansion = "Drug Reaction with Eosinophilia and Systemic Symptoms"
    documents = (
        _document("train-1", f"{expansion} (DRESS).", group_id="group-1"),
        _document("train-2", f"{expansion} (DRESS).", group_id="group-2"),
        _document("dev-1", f"{expansion} (DRESS).", group_id="group-3"),
    )
    result = mine_abbreviations(
        documents,
        (_artifact(),),
        {"train-1": "train", "train-2": "train", "dev-1": "development"},
        _policy(),
    )
    concept = ConceptEntry(
        concept_id="ICD10:D72.12",
        code="D72.12",
        code_system=CodeSystem.ICD10,
        canonical_name=expansion,
        semantic_type=EntityType.DISEASE,
    )

    report = benchmark_abbreviation_knowledge(
        result.definitions,
        (),
        result.abbreviation_table,
        evaluation_splits=("development",),
        repository=_FixtureTerminologyRepository(concept),
    )

    assert report["baseline"]["exact_recall"] == 0.0
    assert report["enriched"]["exact_recall"] == 1.0
    assert report["evaluation_unique_pair_count"] == 1
    assert report["unique_pairs"]["enriched"]["exact_recall"] == 1.0
    assert report["retrieval"]["resolvable_case_count"] == 1
    assert report["retrieval"]["baseline"]["hit_at_1"] == 0.0
    assert report["retrieval"]["enriched"]["hit_at_1"] == 1.0
    assert report["retrieval"]["improved_case_count"] == 1
    assert report["retrieval"]["regressed_case_count"] == 0
    assert report["retrieval"]["cases"][0]["expected_code"] == "D72.12"


def test_policy_loader_requires_source_version_and_disjoint_splits(tmp_path: Path) -> None:
    policy_path = tmp_path / "abbreviations.yaml"
    policy_path.write_text(
        f"""\
schema_version: abbreviation-mining-policy.v1
policy_id: fixture-v1
accepted_source_ids: [fixture]
accepted_source_versions: [{_SOURCE_VERSION}]
accepted_source_sha256: [{_SOURCE_SHA256}]
accepted_languages: [en]
knowledge_splits: [train]
evaluation_splits: [development]
min_supporting_documents: 2
min_supporting_groups: 2
""",
        encoding="utf-8",
    )

    policy = load_abbreviation_mining_policy(policy_path)

    assert policy.accepted_source_versions == (_SOURCE_VERSION,)
    assert policy.knowledge_splits == ("train",)
    assert policy.evaluation_splits == ("development",)


def test_runtime_table_merges_and_normalizes_without_dropping_ambiguity() -> None:
    table = build_runtime_abbreviation_table(
        (
            {"abbreviation": "MI", "expansions": ["myocardial infarction"]},
            {"abbreviation": "mi", "expansions": ["mitral insufficiency"]},
        ),
        (
            {"abbreviation": "MRI", "expansions": ["magnetic resonance imaging"]},
            {"abbreviation": "MI", "expansions": ["Myocardial Infarction"]},
        ),
    )

    assert table == (
        {
            "abbreviation": "MI",
            "normalized_abbreviation": "mi",
            "expansions": ["mitral insufficiency", "myocardial infarction"],
            "normalized_expansions": ["mitral insufficiency", "myocardial infarction"],
            "source_layers": ["baseline", "mined"],
        },
        {
            "abbreviation": "MRI",
            "normalized_abbreviation": "mri",
            "expansions": ["magnetic resonance imaging"],
            "normalized_expansions": ["magnetic resonance imaging"],
            "source_layers": ["mined"],
        },
    )


def test_cli_exposes_reproducible_abbreviation_mining_inputs() -> None:
    args = build_parser().parse_args(
        [
            "data",
            "knowledge",
            "mine-abbreviations",
            "--documents",
            "documents.jsonl",
            "--artifacts",
            "artifacts.jsonl",
            "--split-manifest",
            "snapshot.json",
            "--policy",
            "policy.yaml",
            "--definitions-output",
            "definitions.jsonl",
            "--candidates-output",
            "candidates.jsonl",
            "--table-output",
            "abbreviations.jsonl",
            "--runtime-table-output",
            "runtime-abbreviations.jsonl",
            "--conflicts-output",
            "conflicts.jsonl",
            "--benchmark-output",
            "benchmark.json",
            "--report-output",
            "report.json",
        ]
    )

    assert args.handler == "data_knowledge_mine_abbreviations"
    assert args.retrieval_limit == 20


def _policy(*, minimum_support: int = 2) -> AbbreviationMiningPolicy:
    return AbbreviationMiningPolicy(
        policy_id="fixture-v1",
        accepted_source_ids=("fixture",),
        accepted_source_versions=(_SOURCE_VERSION,),
        accepted_source_sha256=(_SOURCE_SHA256,),
        accepted_languages=("en",),
        knowledge_splits=("train",),
        evaluation_splits=("development",),
        min_supporting_documents=minimum_support,
        min_supporting_groups=minimum_support,
    )


def _artifact() -> SourceArtifact:
    return SourceArtifact(
        artifact_id="fixture:artifact",
        source_id="fixture",
        source_version=_SOURCE_VERSION,
        source_uri="https://example.test/fixture.jsonl",
        object=StoredObject(
            sha256=_SOURCE_SHA256,
            uri=f"medical-kg-cas://sha256/{_SOURCE_SHA256}",
            byte_size=1,
        ),
        media_type="application/jsonl",
        license_id="CC0-1.0",
        access_class=AccessClass.OPEN,
        redistribution=RedistributionPolicy.ALLOWED,
        hosted_processing_allowed=True,
        retrieved_at="2026-07-22T00:00:00Z",
    )


def _document(document_id: str, text: str, *, group_id: str) -> MinedDocument:
    return MinedDocument(
        document_id=document_id,
        text=text,
        language="en",
        note_type="fixture",
        source_artifact_id="fixture:artifact",
        access_class=AccessClass.OPEN,
        redistribution=RedistributionPolicy.ALLOWED,
        hosted_processing_allowed=True,
        group_ids=(group_id,),
    )


class _FixtureTerminologyRepository:
    """Minimal repository used to prove that expansion lookup changes retrieval rank."""

    def __init__(self, concept: ConceptEntry) -> None:
        self._concept = concept

    def get_by_concept_id(self, concept_id: str) -> ConceptEntry | None:
        return self._concept if concept_id == self._concept.concept_id else None

    def get_by_code(self, code_system: CodeSystem, code: str) -> ConceptEntry | None:
        if code_system == self._concept.code_system and code == self._concept.code:
            return self._concept
        return None

    def exact_lookup(
        self,
        mention: str,
        *,
        entity_type: EntityType | None = None,
        code_systems: Sequence[CodeSystem] | None = None,
        limit: int = 20,
    ) -> list[ConceptEntry]:
        if mention.casefold() != self._concept.canonical_name.casefold():
            return []
        if entity_type is not None and entity_type != self._concept.semantic_type:
            return []
        if code_systems is not None and self._concept.code_system not in code_systems:
            return []
        return [self._concept][:limit]

    def toneless_lookup(
        self,
        mention: str,
        *,
        entity_type: EntityType | None = None,
        code_systems: Sequence[CodeSystem] | None = None,
        limit: int = 20,
    ) -> list[ConceptEntry]:
        return self.exact_lookup(
            mention,
            entity_type=entity_type,
            code_systems=code_systems,
            limit=limit,
        )

    def search(
        self,
        mention: str,
        *,
        entity_type: EntityType | None = None,
        code_systems: Sequence[CodeSystem] | None = None,
        limit: int = 20,
    ) -> list[ConceptEntry]:
        return []
