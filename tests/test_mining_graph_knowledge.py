"""Tests for provenance-preserving graph compilation and semantic deduplication."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from medical_kg_nlp.mining.graph_knowledge import (
    GraphCompilationConfig,
    compile_knowledge_graph,
)
from medical_kg_nlp.mining.records import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    ConceptLink,
    MinedDocument,
    RedistributionPolicy,
    RelationProposal,
    ReviewStatus,
)
from medical_kg_nlp.utils.io import read_jsonl, write_jsonl


def _document(document_id: str, text: str) -> MinedDocument:
    return MinedDocument(
        document_id=document_id,
        text=text,
        language="en",
        note_type="drug_label",
        source_artifact_id=f"artifact:{document_id}",
        access_class=AccessClass.OPEN,
        redistribution=RedistributionPolicy.ALLOWED,
        hosted_processing_allowed=True,
    )


def _annotation(
    annotation_id: str,
    document: MinedDocument,
    span: tuple[int, int],
    entity_type: str,
    concept: tuple[str, str] | None,
) -> AnnotationProposal:
    concepts = (
        ()
        if concept is None
        else (
            ConceptLink(
                code_system=concept[0],
                code=concept[1],
                terminology_version="source-v1",
            ),
        )
    )
    return AnnotationProposal(
        annotation_id=annotation_id,
        document_id=document.document_id,
        span=span,
        text=document.text[span[0] : span[1]],
        entity_type=entity_type,
        assertions=(),
        concepts=concepts,
        confidence=1.0,
        layer=AnnotationLayer.SILVER,
        label_source="source_structured_annotation",
        labeler_id="fixture-labeler:v1",
    )


def _relation(
    relation_id: str,
    document: MinedDocument,
    head: AnnotationProposal,
    tail: AnnotationProposal,
    relation_type: str,
) -> RelationProposal:
    return RelationProposal(
        relation_id=relation_id,
        document_id=document.document_id,
        head_annotation_id=head.annotation_id,
        tail_annotation_id=tail.annotation_id,
        relation_type=relation_type,
        confidence=1.0,
        layer=AnnotationLayer.SILVER,
        label_source="source_structured_relation",
        evidence_span=(head.span[0], tail.span[1]),
        labeler_id="fixture-relations:v1",
    )


def test_graph_compiler_deduplicates_concepts_terms_and_edges(tmp_path: Path) -> None:
    terminology = tmp_path / "terminology.jsonl"
    overlay = tmp_path / "aliases.jsonl"
    write_jsonl(
        terminology,
        [
            {
                "concept_id": "ICD10:I00",
                "code": "I00",
                "code_system": "ICD-10",
                "canonical_name": "root disease",
                "semantic_type": "DISEASE",
            },
            {
                "concept_id": "ICD10:I10",
                "code": "I10",
                "code_system": "ICD-10",
                "canonical_name": "hypertension",
                "semantic_type": "DISEASE",
                "parents": ["I00"],
            },
        ],
    )
    write_jsonl(
        overlay,
        [
            {
                "alias": "cao huyết áp",
                "code": "I10",
                "code_system": "ICD-10",
                "semantic_type": "DISEASE",
            },
            {
                "alias": "tăng HA",
                "target_concept_id": "ICD10:I10",
                "semantic_type": "DISEASE",
            },
            {
                "alias": "local only",
                "target_concept_id": "LOCAL:UNKNOWN",
                "semantic_type": "OTHER",
            },
        ],
    )
    documents = (
        _document("doc-1", "Drug A | Drug Alpha | ORAL"),
        _document("doc-2", "drug a | drug alpha | oral"),
    )
    annotations: list[AnnotationProposal] = []
    relations: list[RelationProposal] = []
    for index, document in enumerate(documents, start=1):
        product = _annotation(f"product-{index}", document, (0, 6), "DRUG", ("NDC", "111"))
        generic = _annotation(f"generic-{index}", document, (9, 19), "DRUG", None)
        route = _annotation(f"route-{index}", document, (22, 26), "ROUTE", ("NCI", "C1"))
        annotations.extend((product, generic, route))
        relations.extend(
            (
                _relation(f"generic-rel-{index}", document, product, generic, "HAS_GENERIC_NAME"),
                _relation(f"route-rel-{index}", document, product, route, "HAS_ROUTE"),
            )
        )

    outputs = {
        name: tmp_path / f"{name}.jsonl" for name in ("nodes", "edges", "evidence")
    }
    report_path = tmp_path / "report.json"
    report = compile_knowledge_graph(
        terminology_paths=(terminology,),
        alias_overlay_paths=(overlay,),
        documents=documents,
        annotations=tuple(annotations),
        relations=tuple(relations),
        config=GraphCompilationConfig(),
        nodes_output=outputs["nodes"],
        edges_output=outputs["edges"],
        evidence_output=outputs["evidence"],
        report_output=report_path,
    )

    nodes = read_jsonl(outputs["nodes"])
    edges = read_jsonl(outputs["edges"])
    assert report["output_counts"] == {
        "nodes": 5,
        "concept_nodes": 4,
        "term_nodes": 1,
        "edges": 3,
        "evidence": 5,
    }
    ndc = next(node for node in nodes if node["code_system"] == "NDC")
    generic = next(node for node in nodes if node["kind"] == "TERM")
    assert ndc["occurrence_count"] == 2
    assert ndc["document_count"] == 2
    assert set(ndc["aliases"]) == {"drug a"}
    assert generic["occurrence_count"] == 2
    route_edge = next(edge for edge in edges if edge["relation_type"] == "HAS_ROUTE")
    assert route_edge["support_count"] == 2
    assert route_edge["document_count"] == 2
    hierarchy = next(edge for edge in edges if edge["relation_type"] == "IS_A")
    assert hierarchy["support_count"] == 1
    i10 = next(node for node in nodes if node["code"] == "I10")
    assert {"cao huyết áp", "tăng HA"}.issubset(i10["aliases"])
    assert report["decision_counts"]["unknown_alias_target_skipped"] == 1
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_graph_compiler_skips_relations_with_quality_filtered_endpoints(
    tmp_path: Path,
) -> None:
    document = _document("doc-1", "Drug A | ORAL")
    head = _annotation("head", document, (0, 6), "DRUG", ("NDC", "111"))
    tail = _annotation("tail", document, (9, 13), "ROUTE", ("NCI", "C1"))
    tail = replace(tail, review_status=ReviewStatus.NEEDS_REVIEW)
    report = compile_knowledge_graph(
        terminology_paths=(),
        alias_overlay_paths=(),
        documents=(document,),
        annotations=(head, tail),
        relations=(_relation("rel", document, head, tail, "HAS_ROUTE"),),
        config=GraphCompilationConfig(),
        nodes_output=tmp_path / "nodes.jsonl",
        edges_output=tmp_path / "edges.jsonl",
        evidence_output=tmp_path / "evidence.jsonl",
        report_output=tmp_path / "report.json",
    )

    assert report["decision_counts"]["annotation_review_filter_skipped"] == 1
    assert report["decision_counts"]["relation_filtered_endpoint_skipped"] == 1
    assert report["output_counts"]["edges"] == 0


def test_graph_compiler_endpoint_only_mode_excludes_unrelated_annotations(
    tmp_path: Path,
) -> None:
    document = _document("doc-1", "Drug A | ORAL | unrelated")
    head = _annotation("head", document, (0, 6), "DRUG", ("NDC", "111"))
    tail = _annotation("tail", document, (9, 13), "ROUTE", ("NCI", "C1"))
    unrelated = _annotation(
        "unrelated",
        document,
        (16, 25),
        "DISEASE",
        ("ICD-10", "I10"),
    )

    report = compile_knowledge_graph(
        terminology_paths=(),
        alias_overlay_paths=(),
        documents=(document,),
        annotations=(head, tail, unrelated),
        relations=(_relation("rel", document, head, tail, "HAS_ROUTE"),),
        config=GraphCompilationConfig(relation_endpoints_only=True),
        nodes_output=tmp_path / "nodes.jsonl",
        edges_output=tmp_path / "edges.jsonl",
        evidence_output=tmp_path / "evidence.jsonl",
        report_output=tmp_path / "report.json",
    )

    nodes = read_jsonl(tmp_path / "nodes.jsonl")
    assert {node["code"] for node in nodes} == {"111", "C1"}
    assert report["decision_counts"]["annotation_non_endpoint_skipped"] == 1
    assert report["config"]["relation_endpoints_only"] is True


def test_graph_compiler_can_reject_codes_absent_from_canonical_terminology(
    tmp_path: Path,
) -> None:
    document = _document("doc-1", "Drug A | ORAL")
    head = _annotation("head", document, (0, 6), "DRUG", ("NDC", "111"))
    tail = _annotation("tail", document, (9, 13), "ROUTE", ("NCI", "C1"))

    report = compile_knowledge_graph(
        terminology_paths=(),
        alias_overlay_paths=(),
        documents=(document,),
        annotations=(head, tail),
        relations=(_relation("rel", document, head, tail, "HAS_ROUTE"),),
        config=GraphCompilationConfig(require_canonical_concepts=True),
        nodes_output=tmp_path / "nodes.jsonl",
        edges_output=tmp_path / "edges.jsonl",
        evidence_output=tmp_path / "evidence.jsonl",
        report_output=tmp_path / "report.json",
    )

    assert report["output_counts"]["nodes"] == 0
    assert report["output_counts"]["edges"] == 0
    assert report["decision_counts"] == {
        "annotation_unknown_canonical_concept_skipped": 2,
        "relation_filtered_endpoint_skipped": 1,
    }


def test_graph_compiler_promotes_unique_rxnorm_attributes(tmp_path: Path) -> None:
    terminology = tmp_path / "rxnorm.jsonl"
    write_jsonl(
        terminology,
        [
            {
                "concept_id": "RXNORM:1",
                "code": "1",
                "code_system": "RxNorm",
                "canonical_name": "mesna",
                "semantic_type": "DRUG",
                "rxnorm_tty": "IN",
                "ingredient": "mesna",
            },
            {
                "concept_id": "RXNORM:2",
                "code": "2",
                "code_system": "RxNorm",
                "canonical_name": "mesna 100 MG Oral Tablet",
                "semantic_type": "DRUG",
                "rxnorm_tty": "SCD",
                "ingredient": "mesna",
                "dose_form": "Oral Tablet",
                "strength": "RXN_AVAILABLE_STRENGTH=100 MG",
            },
            {
                "concept_id": "RXNORM:3",
                "code": "3",
                "code_system": "RxNorm",
                "canonical_name": "Mesnex",
                "semantic_type": "DRUG",
                "rxnorm_tty": "BN",
                "ingredient": "mesna",
            },
        ],
    )
    report = compile_knowledge_graph(
        terminology_paths=(terminology,),
        alias_overlay_paths=(),
        documents=(),
        annotations=(),
        relations=(),
        config=GraphCompilationConfig(),
        nodes_output=tmp_path / "nodes.jsonl",
        edges_output=tmp_path / "edges.jsonl",
        evidence_output=tmp_path / "evidence.jsonl",
        report_output=tmp_path / "report.json",
    )
    edges = read_jsonl(tmp_path / "edges.jsonl")
    assert report["decision_counts"]["structured_ingredient_observation_count"] == 2
    assert report["decision_counts"]["structured_dose_form_observation_count"] == 1
    assert report["decision_counts"]["structured_strength_observation_count"] == 1
    assert {edge["relation_type"] for edge in edges} == {
        "HAS_ACTIVE_INGREDIENT",
        "HAS_DOSAGE_FORM",
        "HAS_STRENGTH",
    }
