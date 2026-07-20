"""Source-aware sentence co-occurrence mining regression tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from medical_kg_nlp.cli.parser import build_parser
from medical_kg_nlp.mining.cooccurrence import (
    CooccurrenceMiningPolicy,
    load_cooccurrence_policy,
    mine_cooccurrence_relations,
)
from medical_kg_nlp.mining.records import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    ConceptLink,
    MinedDocument,
    RedistributionPolicy,
    ReviewStatus,
)


def _document(
    document_id: str,
    text: str,
    *,
    corpus_split: str = "train",
) -> MinedDocument:
    return MinedDocument(
        document_id=document_id,
        text=text,
        language="es",
        note_type="clinical_case",
        source_artifact_id="codiesp:fixture",
        access_class=AccessClass.OPEN,
        redistribution=RedistributionPolicy.ATTRIBUTION,
        hosted_processing_allowed=True,
        metadata={"parser_id": "codiesp", "corpus_split": corpus_split},
    )


def _annotation(
    document: MinedDocument,
    annotation_id: str,
    text: str,
    entity_type: str,
    code: str,
) -> AnnotationProposal:
    start = document.text.index(text)
    return AnnotationProposal(
        annotation_id=annotation_id,
        document_id=document.document_id,
        span=(start, start + len(text)),
        text=text,
        entity_type=entity_type,
        assertions=(),
        concepts=(ConceptLink("ICD-10", code, "TT06-2026-BYT"),),
        confidence=1.0,
        layer=AnnotationLayer.SILVER,
        label_source="source_human_annotation",
        labeler_id="fixture",
        review_status=ReviewStatus.PROPOSED,
        metadata={"discontinuous": "false", "source_text_match": "true"},
    )


def _policy() -> CooccurrenceMiningPolicy:
    return CooccurrenceMiningPolicy(
        policy_id="codiesp-train-cooccurrence-v1",
        relation_type="CO_OCCURS_WITH",
        accepted_source_ids=("codiesp",),
        document_metadata_filters=(("corpus_split", ("train",)),),
        accepted_layers=(AnnotationLayer.SILVER,),
        accepted_review_statuses=(ReviewStatus.PROPOSED,),
        allowed_entity_pairs=(("DISEASE", "PROCEDURE"),),
        minimum_documents=2,
    )


def test_cooccurrence_requires_multi_document_same_sentence_support() -> None:
    first = _document("codiesp:1", "cirrosis requirio trasplante.")
    second = _document("codiesp:2", "trasplante indicado por cirrosis.")
    annotations = (
        _annotation(first, "a-disease-1", "cirrosis", "DISEASE", "K74.6"),
        _annotation(first, "a-procedure-1", "trasplante", "PROCEDURE", "Z94.4"),
        _annotation(second, "a-procedure-2", "trasplante", "PROCEDURE", "Z94.4"),
        _annotation(second, "a-disease-2", "cirrosis", "DISEASE", "K74.6"),
    )

    result = mine_cooccurrence_relations((first, second), annotations, _policy())

    assert len(result.relations) == 2
    assert {relation.relation_type for relation in result.relations} == {
        "CO_OCCURS_WITH"
    }
    # The semantic ordering is stable even when the mentions appear in reverse text order.
    assert {relation.head_annotation_id for relation in result.relations} == {
        "a-disease-1",
        "a-disease-2",
    }
    for relation in result.relations:
        document = first if relation.document_id == first.document_id else second
        relation.validate(document, {item.annotation_id: item for item in annotations})
        assert relation.layer is AnnotationLayer.BRONZE
        assert relation.metadata["semantic_inference"] == "false"
        assert relation.metadata["support_document_count"] == "2"


def test_cooccurrence_rejects_cross_sentence_and_official_dev_records() -> None:
    train = _document("codiesp:train", "cirrosis. trasplante posterior.")
    dev = _document(
        "codiesp:dev",
        "cirrosis requirio trasplante.",
        corpus_split="dev",
    )
    annotations = (
        _annotation(train, "train-disease", "cirrosis", "DISEASE", "K74.6"),
        _annotation(train, "train-procedure", "trasplante", "PROCEDURE", "Z94.4"),
        _annotation(dev, "dev-disease", "cirrosis", "DISEASE", "K74.6"),
        _annotation(dev, "dev-procedure", "trasplante", "PROCEDURE", "Z94.4"),
    )

    result = mine_cooccurrence_relations(
        (train, dev),
        annotations,
        replace(_policy(), minimum_documents=1),
    )

    assert not result.relations
    assert result.report["counters"]["documents_rejected:metadata"] == 1
    assert result.report["counters"]["candidate_occurrences"] == 0


def test_cooccurrence_support_threshold_counts_documents_not_mentions() -> None:
    document = _document(
        "codiesp:repeat",
        "cirrosis requirio trasplante. cirrosis tras trasplante.",
    )
    first_disease = _annotation(
        document, "disease-1", "cirrosis", "DISEASE", "K74.6"
    )
    first_procedure = _annotation(
        document, "procedure-1", "trasplante", "PROCEDURE", "Z94.4"
    )
    second_disease_start = document.text.rindex("cirrosis")
    second_procedure_start = document.text.rindex("trasplante")
    annotations = (
        first_disease,
        first_procedure,
        replace(
            first_disease,
            annotation_id="disease-2",
            span=(second_disease_start, second_disease_start + len("cirrosis")),
        ),
        replace(
            first_procedure,
            annotation_id="procedure-2",
            span=(second_procedure_start, second_procedure_start + len("trasplante")),
        ),
    )

    result = mine_cooccurrence_relations((document,), annotations, _policy())

    assert not result.relations
    assert result.report["counters"]["unsupported_semantic_pairs"] == 1


def test_cooccurrence_skips_dense_sentences_before_quadratic_pairing() -> None:
    document = _document("codiesp:dense", "cirrosis trasplante hepatitis biopsia.")
    annotations = (
        _annotation(document, "a1", "cirrosis", "DISEASE", "K74.6"),
        _annotation(document, "a2", "trasplante", "PROCEDURE", "Z94.4"),
        _annotation(document, "a3", "hepatitis", "DISEASE", "B19.9"),
        _annotation(document, "a4", "biopsia", "PROCEDURE", "Z01.8"),
    )
    policy = replace(
        _policy(), minimum_documents=1, max_annotations_per_sentence=3
    )

    result = mine_cooccurrence_relations((document,), annotations, policy)

    assert not result.relations
    assert result.report["counters"]["sentence_rejected:annotation_density"] == 1


def test_source_block_scope_connects_bounded_cross_sentence_evidence() -> None:
    document = _document(
        "codiesp:block",
        "wilson disease was confirmed. tremor was observed.",
    )
    block_span = f"[0,{len(document.text)}]"
    disease = replace(
        _annotation(
            document,
            "block-disease",
            "wilson disease",
            "DISEASE",
            "E83.01",
        ),
        metadata={"source_block_span": block_span},
    )
    symptom = replace(
        _annotation(document, "block-symptom", "tremor", "SYMPTOM", "R25.1"),
        metadata={"source_block_span": block_span},
    )
    policy = replace(
        _policy(),
        allowed_entity_pairs=(("DISEASE", "SYMPTOM"),),
        context_scope="source_block",
        minimum_documents=1,
    )

    result = mine_cooccurrence_relations((document,), (disease, symptom), policy)

    assert len(result.relations) == 1
    assert result.relations[0].evidence_span == (0, len(document.text))
    assert result.relations[0].metadata["context_scope"] == "source_block"
    assert result.report["semantic_contract"] == (
        "same_source_block_observation_without_causal_inference"
    )


def test_cooccurrence_selects_preferred_cross_system_links_and_rejects_context() -> None:
    document = _document("codiesp:ontology", "wilson disease caused tremor.")
    disease = replace(
        _annotation(
            document,
            "disease",
            "wilson disease",
            "DISEASE",
            "E83.01",
        ),
        concepts=(
            ConceptLink("ICD-10", "E83.01", "TT06-2026-BYT"),
            ConceptLink("MONDO", "0010200", "mondo:2026-07-06"),
        ),
    )
    symptom = replace(
        _annotation(document, "symptom", "tremor", "SYMPTOM", "R25.1"),
        concepts=(
            ConceptLink("LOCAL", "symptom:tremor", "local:v1"),
            ConceptLink("HPO", "0001337", "hpo:2026-06-23"),
        ),
    )
    policy = replace(
        _policy(),
        allowed_entity_pairs=(("DISEASE", "SYMPTOM"),),
        preferred_code_systems_by_entity_type=(
            ("DISEASE", ("MONDO",)),
            ("SYMPTOM", ("HPO",)),
        ),
        rejected_assertions=("NEGATED", "FAMILY", "POSSIBLE"),
        minimum_documents=1,
    )

    accepted = mine_cooccurrence_relations(
        (document,), (disease, symptom), policy
    )
    rejected = mine_cooccurrence_relations(
        (document,),
        (disease, replace(symptom, assertions=("NEGATED",))),
        policy,
    )

    assert len(accepted.relations) == 1
    pair = accepted.report["top_supported_pairs"][0]
    assert {
        (pair[endpoint]["entity_type"], pair[endpoint]["code_system"])
        for endpoint in ("head", "tail")
    } == {("DISEASE", "MONDO"), ("SYMPTOM", "HPO")}
    assert not rejected.relations
    assert rejected.report["counters"]["annotations_rejected:assertion"] == 1


def test_cooccurrence_policy_and_cli_are_discoverable() -> None:
    policy = load_cooccurrence_policy(
        "configs/mining/relations/codiesp-train-cooccurrence.yaml"
    )
    args = build_parser().parse_args(
        [
            "data",
            "relation",
            "mine-cooccurrence",
            "--documents",
            "documents.jsonl",
            "--annotations",
            "annotations.jsonl",
            "--policy",
            "policy.yaml",
            "--output",
            "relations.jsonl",
            "--report-output",
            "report.json",
        ]
    )

    assert policy.relation_type == "CO_OCCURS_WITH"
    assert args.handler == "data_relation_mine_cooccurrence"


def test_cooccurrence_policy_rejects_unknown_schema_values() -> None:
    with pytest.raises(ValueError, match="NOT-A-SYSTEM"):
        replace(
            _policy(),
            preferred_code_systems_by_entity_type=(
                ("DISEASE", ("NOT-A-SYSTEM",)),
            ),
        )
    with pytest.raises(ValueError, match="NOT-AN-ASSERTION"):
        replace(_policy(), rejected_assertions=("NOT-AN-ASSERTION",))


def test_cooccurrence_policy_loader_rejects_truthy_boolean_strings(
    tmp_path: Path,
) -> None:
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        """\
schema_version: medical-cooccurrence-policy.v1
policy_id: invalid-boolean
relation_type: CO_OCCURS_WITH
accepted_source_ids: [codiesp]
accepted_layers: [silver]
accepted_review_statuses: [proposed]
allowed_entity_pairs: [[DISEASE, PROCEDURE]]
require_contiguous: "false"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="require_contiguous.*boolean"):
        load_cooccurrence_policy(policy_path)
