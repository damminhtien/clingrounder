from __future__ import annotations

import json
from pathlib import Path

from clingrounder.retrieval.adapters import FTSRetrieverAdapter
from clingrounder.retrieval.query_expansion import build_retrieval_query_variants
from clingrounder.schema.types import EntityType
from clingrounder.terminology import SQLiteTerminologyRepository, build_terminology_index


def test_medication_query_expansion_is_bounded_and_type_specific() -> None:
    variants = build_retrieval_query_variants("IV aspirin 325mg once", EntityType.DRUG)

    assert [(variant.text, variant.kind) for variant in variants] == [
        ("iv aspirin 325 mg once", "medication_typography"),
        ("aspirin", "medication_core"),
    ]
    assert build_retrieval_query_variants(
        "đái tháo đường type 2",
        EntityType.DISEASE,
    ) == ()
    oral_variants = build_retrieval_query_variants(
        "furosemide 40 mg đường uống",
        EntityType.DRUG,
    )
    assert [variant.text for variant in oral_variants] == [
        "furosemide 40 mg oral",
        "furosemide",
    ]


def test_fts_query_expansion_recovers_attached_rxnorm_strength(tmp_path: Path) -> None:
    repository = _repository(
        tmp_path,
        [
            _drug("RX:1191", "1191", "aspirin", tty="IN"),
            _drug(
                "RX:212033",
                "212033",
                "aspirin 325 MG Oral Tablet",
                tty="SCD",
                ingredient="aspirin",
                strength="325 MG",
                dose_form="Oral Tablet",
            ),
        ],
    )

    candidates = FTSRetrieverAdapter(repository).retrieve(
        "aspirin 325mg",
        EntityType.DRUG,
        "",
        20,
    )

    by_code = {candidate.code: candidate for candidate in candidates}
    assert "212033" in by_code
    assert candidates[0].code == "212033"


def test_fts_query_expansion_recovers_ingredient_from_route_and_sig(tmp_path: Path) -> None:
    repository = _repository(
        tmp_path,
        [_drug("RX:7052", "7052", "morphine", tty="IN")],
    )

    candidates = FTSRetrieverAdapter(repository).retrieve(
        "IV morphine 4 mg once",
        EntityType.DRUG,
        "",
        20,
    )

    assert [candidate.code for candidate in candidates] == ["7052"]
    assert candidates[0].score == 0.65


def _repository(
    tmp_path: Path,
    rows: list[dict[str, object]],
) -> SQLiteTerminologyRepository:
    source = tmp_path / "rxnorm.jsonl"
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = build_terminology_index((source,), cache_dir=tmp_path / "cache")
    return SQLiteTerminologyRepository(manifest.index_path)


def _drug(
    concept_id: str,
    code: str,
    canonical_name: str,
    *,
    tty: str,
    ingredient: str | None = None,
    strength: str | None = None,
    dose_form: str | None = None,
) -> dict[str, object]:
    return {
        "concept_id": concept_id,
        "code": code,
        "code_system": "RxNorm",
        "canonical_name": canonical_name,
        "semantic_type": "DRUG",
        "rxnorm_tty": tty,
        "ingredient": ingredient,
        "strength": strength,
        "dose_form": dose_form,
        "source": "fixture",
    }
