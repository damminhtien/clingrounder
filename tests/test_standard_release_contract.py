"""Runtime terminology profiles must stay aligned with pinned source releases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.terminology import (
    CompositeTerminologyRepository,
    InMemoryTerminologyRepository,
)


def test_runtime_source_versions_match_tt06_and_rxnorm_import_manifests() -> None:
    versions = _load_json(Path("data/standards/source_versions.json"))
    tt06_manifest = _load_json(
        Path("data/standards/icd10_vn/processed/tt06_icd10_import_manifest.json")
    )
    rxnorm_manifest = _load_json(
        Path(
            "data/standards/rxnorm/processed/"
            "rxnorm_full_07062026_import_manifest.json"
        )
    )

    icd_source = versions["icd10_vn"]
    assert icd_source["source_id"] == (
        tt06_manifest["source_policy"]["primary_source"]["source_id"]
    )
    assert icd_source["effective_date"] == (
        tt06_manifest["source_policy"]["primary_source"]["effective_date"]
    )
    assert tt06_manifest["concepts"] == 14925

    rxnorm_source = versions["rxnorm"]
    imported_source = rxnorm_manifest["source_policy"]["source"]
    assert rxnorm_source["release_date"] == imported_source["release_date"]
    assert rxnorm_source["fallback_source_id"] == imported_source["fallback_source_id"]
    assert rxnorm_source["fallback_file"].endswith(
        "rxnorm_full_07062026_concepts.jsonl"
    )
    assert rxnorm_manifest["concepts"] == 73912


def test_full_profiles_query_current_full_tt06_and_rxnorm_without_sample_memory() -> None:
    for path in (
        Path("configs/phase1_full.yaml"),
        Path("configs/pipeline/full_terminology.yaml"),
    ):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        terminology = payload["terminology"]
        normalization_paths = {
            Path(value).name for value in terminology["normalization_paths"]
        }

        assert normalization_paths == {
            "tt06_icd10_concepts.jsonl",
            "rxnorm_full_07062026_concepts.jsonl",
        }
        assert terminology.get("reviewed_mention_path") is None


def test_current_rxnorm_contains_july_only_concept_not_present_in_june_snapshot() -> None:
    july = Path(
        "data/standards/rxnorm/processed/rxnorm_prescribable_07062026_concepts.jsonl"
    )
    june = Path(
        "data/standards/rxnorm/processed/rxnorm_prescribable_06012026_concepts.jsonl"
    )

    current = _find_code(july, "2743603")

    assert current is not None
    assert current["canonical_name"] == "bulevirtide"
    assert current["source"] == "rxnorm_prescribable_2026_07_06"
    assert _find_code(june, "2743603") is None


@pytest.mark.release
def test_runtime_repository_queries_current_tt06_and_july_rxnorm_releases() -> None:
    """Exercise the typed query contract on concepts absent from legacy seed/release data."""

    repository = CompositeTerminologyRepository(
        (
            InMemoryTerminologyRepository(
                DictionaryStore.from_jsonl(
                    "data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl"
                )
            ),
            InMemoryTerminologyRepository(
                DictionaryStore.from_jsonl(
                    "data/standards/rxnorm/processed/rxnorm_full_07062026_concepts.jsonl"
                )
            ),
        )
    )

    icd = repository.exact_lookup(
        "Bệnh tả do vi khuẩn Vibrio cholerae 01, típ sinh học cholerae",
        entity_type=EntityType.DISEASE,
        code_systems=(CodeSystem.ICD10,),
        limit=5,
    )
    rxnorm = repository.exact_lookup(
        "bulevirtide",
        entity_type=EntityType.DRUG,
        code_systems=(CodeSystem.RXNORM,),
        limit=5,
    )

    assert [(entry.code, entry.source) for entry in icd] == [
        ("A00.0", "icd10_vn_tt06_2026")
    ]
    # A bare ingredient can also be an alias of a semantic clinical drug form. The repository
    # must preserve that current-release ambiguity so assignment can use TTY/structured context
    # instead of treating normalized string equality as a unique RxCUI.
    assert [(entry.code, entry.rxnorm_tty) for entry in rxnorm] == [
        ("2743603", "IN"),
        ("2743608", "SCDF"),
    ]
    assert {entry.source for entry in rxnorm} == {"rxnorm_full_2026_07_06"}


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _find_code(path: Path, code: str) -> dict[str, object] | None:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("code") == code:
                return row
    return None
