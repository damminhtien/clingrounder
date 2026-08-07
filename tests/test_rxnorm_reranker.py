"""Production contracts for structured RxNorm candidate reranking."""

from __future__ import annotations

from clingrounder.dictionaries.dictionary_store import DictionaryStore
from clingrounder.dictionaries.synonym_table import ConceptEntry
from clingrounder.linking.candidate import Candidate
from clingrounder.linking.rxnorm_reranker import StructuredRxNormReranker
from clingrounder.schema.types import CodeSystem, EntityType
from clingrounder.terminology.memory import InMemoryTerminologyRepository


def test_clonazepam_product_strength_rejects_nonmatching_candidate() -> None:
    reranker = _reranker(
        _drug("0.5", "clonazepam 0.5 MG Oral Tablet"),
        _drug("1", "clonazepam 1 MG Oral Tablet"),
    )

    ranked = reranker.rerank(
        [_candidate("0.5"), _candidate("1")],
        mention="clonazepam 0.5 mg oral tablet",
    )

    assert [candidate.code for candidate in ranked] == ["0.5"]


def test_release_mismatch_rejects_immediate_release_metformin() -> None:
    reranker = _reranker(
        _drug(
            "xr",
            "metformin 500 MG Extended Release Oral Tablet",
            ingredient="metformin",
        ),
        _drug(
            "ir",
            "metformin 500 MG Immediate Release Oral Tablet",
            ingredient="metformin",
        ),
    )

    ranked = reranker.rerank(
        [_candidate("ir"), _candidate("xr")],
        mention="metformin XR 500 mg tablet",
    )

    assert [candidate.code for candidate in ranked] == ["xr"]


def test_explicit_500_mg_tablet_rejects_other_product_strength() -> None:
    reranker = _reranker(
        _drug("500", "metformin 500 MG Oral Tablet", ingredient="metformin"),
        _drug("850", "metformin 850 MG Oral Tablet", ingredient="metformin"),
    )

    ranked = reranker.rerank(
        [_candidate("850"), _candidate("500")],
        mention="metformin 500 mg tablet",
    )

    assert [candidate.code for candidate in ranked] == ["500"]


def test_combination_ingredient_rejects_single_ingredient_candidate() -> None:
    reranker = _reranker(
        _drug("amoxicillin", "amoxicillin 500 MG Oral Capsule", ingredient="amoxicillin"),
        _drug(
            "combo",
            "amoxicillin 875 MG / clavulanate 125 MG Oral Tablet",
            ingredient="amoxicillin / clavulanate",
        ),
    )

    ranked = reranker.rerank(
        [_candidate("amoxicillin"), _candidate("combo")],
        mention="amoxicillin/clavulanate 875 mg / 125 mg tablet",
    )

    assert [candidate.code for candidate in ranked] == ["combo"]


def test_brand_and_product_strength_prioritize_viagra_25_mg() -> None:
    reranker = _reranker(
        _drug("25", "sildenafil 25 MG Oral Tablet", ingredient="sildenafil", brand="Viagra"),
        _drug("50", "sildenafil 50 MG Oral Tablet", ingredient="sildenafil", brand="Viagra"),
    )

    ranked = reranker.rerank(
        [_candidate("50"), _candidate("25")],
        mention="sildenafil [Viagra] 25 mg tablet",
    )

    assert [candidate.code for candidate in ranked] == ["25"]


def test_administered_dose_is_soft_ranking_evidence_not_hard_rejection() -> None:
    reranker = _reranker(
        _drug("0.5", "clonazepam 0.5 MG Oral Tablet"),
        _drug("1", "clonazepam 1 MG Oral Tablet"),
    )

    ranked = reranker.rerank(
        [_candidate("1"), _candidate("0.5")],
        mention="clonazepam 1.5 mg po qhs",
    )

    assert {candidate.code for candidate in ranked} == {"0.5", "1"}


def test_intravenous_route_is_not_a_dose_form_conflict() -> None:
    reranker = _reranker(_drug("25", "metoprolol 25 MG Oral Tablet"))

    ranked = reranker.rerank(
        [_candidate("25")],
        mention="metoprolol 25 mg IV",
    )

    assert [candidate.code for candidate in ranked] == ["25"]


def _reranker(*entries: ConceptEntry) -> StructuredRxNormReranker:
    return StructuredRxNormReranker(
        InMemoryTerminologyRepository(DictionaryStore(list(entries)))
    )


def _drug(
    code: str,
    name: str,
    *,
    ingredient: str | None = "clonazepam",
    brand: str | None = None,
) -> ConceptEntry:
    return ConceptEntry(
        concept_id=f"RXNORM:{code}",
        code=code,
        code_system=CodeSystem.RXNORM,
        canonical_name=name,
        semantic_type=EntityType.DRUG,
        ingredient=ingredient,
        brand_name=brand,
        rxnorm_tty="SCD",
    )


def _candidate(code: str) -> Candidate:
    return Candidate(
        concept_id=f"RXNORM:{code}",
        code=code,
        code_system=CodeSystem.RXNORM,
        canonical_name=code,
        semantic_type=EntityType.DRUG,
        score=0.9,
        source="exact",
    )
