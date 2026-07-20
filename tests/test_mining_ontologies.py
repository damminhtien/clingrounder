from __future__ import annotations

import json
from pathlib import Path

import pytest

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.mining.ontologies import (
    OBOGraphCompilationConfig,
    compile_obo_graph_release,
)
from medical_kg_nlp.schema.types import CodeSystem, EntityType


def test_compile_obo_graph_preserves_metadata_and_filters_hierarchy(tmp_path: Path) -> None:
    source = tmp_path / "mondo.json"
    source.write_text(json.dumps(_obo_fixture()), encoding="utf-8")

    report = compile_obo_graph_release(
        input_path=source,
        output_dir=tmp_path / "compiled",
        config=OBOGraphCompilationConfig(
            source_id="mondo",
            source_version="test-release",
            iri_prefix="MONDO",
            code_system=CodeSystem.MONDO,
            entity_type=EntityType.DISEASE,
        ),
    )

    concepts = _read_jsonl(tmp_path / "compiled" / "concepts.jsonl")
    terminology = _read_jsonl(tmp_path / "compiled" / "terminology.jsonl")
    edges = _read_jsonl(tmp_path / "compiled" / "edges.jsonl")
    assert len(concepts) == 3
    assert len(terminology) == 2
    assert len(edges) == 1
    assert concepts[1]["parents"] == ["MONDO:0000001"]
    assert concepts[1]["synonyms"][0]["scope"] == "BROAD"
    assert concepts[2]["deprecated"] is True
    assert concepts[2]["replacement_ids"] == [
        "http://purl.obolibrary.org/obo/MONDO_0000002"
    ]
    assert report["counts"]["deprecated_node_count"] == 1
    assert report["counts"]["inactive_hierarchy_edge_count"] == 1
    assert report["promotion"]["runtime_default"] is False

    store = DictionaryStore.from_jsonl(tmp_path / "compiled" / "terminology.jsonl")
    assert store.exact_lookup("broad child name")[0].code_system == CodeSystem.MONDO


def test_compile_obo_graph_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "hp.json"
    source.write_text(json.dumps(_obo_fixture(prefix="HP")), encoding="utf-8")
    config = OBOGraphCompilationConfig(
        source_id="hpo",
        source_version="test-release",
        iri_prefix="HP",
        code_system=CodeSystem.HPO,
        entity_type=EntityType.FINDING,
    )

    first = compile_obo_graph_release(
        input_path=source,
        output_dir=tmp_path / "first",
        config=config,
    )
    second = compile_obo_graph_release(
        input_path=source,
        output_dir=tmp_path / "second",
        config=config,
    )

    for artifact in ("concepts", "terminology", "nodes", "edges", "evidence"):
        assert first["outputs"][artifact]["sha256"] == second["outputs"][artifact]["sha256"]


def test_obo_graph_config_rejects_invalid_code_type_pair() -> None:
    with pytest.raises(ValueError, match="MONDO is invalid for DRUG"):
        OBOGraphCompilationConfig(
            source_id="mondo",
            source_version="test-release",
            iri_prefix="MONDO",
            code_system=CodeSystem.MONDO,
            entity_type=EntityType.DRUG,
        )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _obo_fixture(prefix: str = "MONDO") -> dict[str, object]:
    iri = f"http://purl.obolibrary.org/obo/{prefix}_"
    return {
        "graphs": [
            {
                "nodes": [
                    {
                        "id": f"{iri}0000001",
                        "lbl": "root disease",
                        "type": "CLASS",
                        "meta": {
                            "definition": {"val": "Root definition", "xrefs": ["PMID:1"]},
                            "xrefs": [{"val": "OMIM:1"}],
                            "synonyms": [
                                {"pred": "hasExactSynonym", "val": "root disorder"}
                            ],
                        },
                    },
                    {
                        "id": f"{iri}0000002",
                        "lbl": "child disease",
                        "type": "CLASS",
                        "meta": {
                            "synonyms": [
                                {"pred": "hasBroadSynonym", "val": "broad child name"}
                            ]
                        },
                    },
                    {
                        "id": f"{iri}0000003",
                        "lbl": "obsolete child",
                        "type": "CLASS",
                        "meta": {
                            "deprecated": True,
                            "basicPropertyValues": [
                                {
                                    "pred": "http://purl.obolibrary.org/obo/IAO_0100001",
                                    "val": f"{iri}0000002",
                                }
                            ],
                        },
                    },
                    {
                        "id": "http://purl.obolibrary.org/obo/OTHER_0001",
                        "lbl": "imported node",
                    },
                ],
                "edges": [
                    {
                        "sub": f"{iri}0000002",
                        "pred": "is_a",
                        "obj": f"{iri}0000001",
                    },
                    {
                        "sub": f"{iri}0000003",
                        "pred": "is_a",
                        "obj": f"{iri}0000001",
                    },
                    {
                        "sub": "http://purl.obolibrary.org/obo/OTHER_0001",
                        "pred": "is_a",
                        "obj": f"{iri}0000001",
                    },
                ],
            }
        ]
    }
