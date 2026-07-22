"""Cross-source corpus fusion and immutable artifact tests."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from medical_kg_nlp.cli.main import main
from medical_kg_nlp.mining.dedup import DuplicateGroupKind, StableTextDeduplicator
from medical_kg_nlp.mining.fusion import CorpusPartition, fuse_corpora
from medical_kg_nlp.mining.io import write_jsonl
from medical_kg_nlp.mining.records import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    MinedDocument,
    RedistributionPolicy,
    RelationProposal,
    ReviewStatus,
)


def _document(
    document_id: str,
    text: str,
    *,
    access_class: AccessClass = AccessClass.OPEN,
    redistribution: RedistributionPolicy = RedistributionPolicy.ALLOWED,
) -> MinedDocument:
    return MinedDocument(
        document_id=document_id,
        text=text,
        language="vi",
        note_type="clinical_note",
        source_artifact_id=f"fixture:{document_id}",
        access_class=access_class,
        redistribution=redistribution,
        hosted_processing_allowed=True,
        metadata={"source_id": document_id.split(":", 1)[0]},
    )


def _annotation(
    document: MinedDocument,
    annotation_id: str,
    span: tuple[int, int],
    entity_type: str,
) -> AnnotationProposal:
    return AnnotationProposal(
        annotation_id=annotation_id,
        document_id=document.document_id,
        span=span,
        text=document.text[span[0] : span[1]],
        entity_type=entity_type,
        assertions=(),
        concepts=(),
        confidence=1.0,
        layer=AnnotationLayer.SILVER,
        label_source="fixture",
        labeler_id="fixture:v1",
        review_status=ReviewStatus.PROPOSED,
    )


def test_fusion_collapses_only_raw_text_and_preserves_relation_audit() -> None:
    left = _document("left:doc", "Sốt và ho.")
    right = _document(
        "right:doc",
        "Sốt và ho.",
        access_class=AccessClass.OPEN_WITH_TERMS,
        redistribution=RedistributionPolicy.ATTRIBUTION,
    )
    fever_left = _annotation(left, "left:fever", (0, 3), "SYMPTOM")
    fever_right = _annotation(right, "right:fever", (0, 3), "SYMPTOM")
    cough_left = _annotation(left, "left:cough", (7, 9), "SYMPTOM")

    drug = _document("drug:doc", "aspirin gây chảy máu")
    aspirin = _annotation(drug, "drug:aspirin", (0, 7), "DRUG")
    bleeding = _annotation(drug, "drug:bleeding", (12, 20), "SYMPTOM")
    relation = RelationProposal(
        relation_id="drug:relation",
        document_id=drug.document_id,
        head_annotation_id=aspirin.annotation_id,
        tail_annotation_id=bleeding.annotation_id,
        relation_type="CAUSES",
        confidence=1.0,
        layer=AnnotationLayer.SILVER,
        label_source="fixture",
    )
    near_left = _document(
        "near:left", "đau đầu dữ dội kéo dài " * 5 + "ba ngày"
    )
    near_right = _document(
        "near:right", "đau đầu dữ dội kéo dài " * 5 + "bốn ngày"
    )

    result = fuse_corpora(
        (
            CorpusPartition(
                "left",
                (left, near_left),
                (fever_left, cough_left),
            ),
            CorpusPartition("right", (right, near_right), (fever_right,)),
            CorpusPartition("drug", (drug,), (aspirin, bleeding), (relation,)),
        ),
        deduplicator=StableTextDeduplicator(hamming_threshold=16),
    )

    assert len(result.documents) == 4
    assert [annotation.text for annotation in result.review_annotations] == ["ho"]
    assert {annotation.text for annotation in result.annotations} == {
        "Sốt",
        "aspirin",
        "chảy máu",
    }
    assert result.relations == (relation,)
    assert result.rejected_relations == ()
    canonical = next(document for document in result.documents if document.text == "Sốt và ho.")
    assert canonical.access_class is AccessClass.OPEN_WITH_TERMS
    assert canonical.redistribution is RedistributionPolicy.ATTRIBUTION
    near_group = next(
        group
        for group in result.duplicate_groups
        if set(group.document_ids) == {"near:left", "near:right"}
    )
    assert near_group.kind is DuplicateGroupKind.NEAR
    assert all(
        near_group.group_id in document.group_ids
        for document in result.documents
        if document.document_id.startswith("near:")
    )


def test_fusion_cli_is_content_addressed_and_idempotent(
    tmp_path: Path, capsys
) -> None:
    document = _document("fixture:doc", "Tăng huyết áp")
    documents_path = tmp_path / "documents.jsonl"
    write_jsonl(documents_path, (document.to_dict(),))
    plan_path = tmp_path / "fusion.yaml"
    plan_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "medical-corpus-fusion-plan.v1",
                "output_root": "fused",
                "run_label": "fixture",
                "sources": [
                    {
                        "source_id": "fixture",
                        "documents": "documents.jsonl",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    first_code = main(["data", "dataset", "fuse", "--plan", str(plan_path)])
    first = json.loads(capsys.readouterr().out)
    second_code = main(["data", "dataset", "fuse", "--plan", str(plan_path)])
    second = json.loads(capsys.readouterr().out)

    assert first_code == second_code == 0
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert first["run_id"] == second["run_id"]
    output_dir = Path(first["output_dir"])
    assert (output_dir / "documents.jsonl").is_file()
    assert (output_dir / "duplicate_groups.jsonl").is_file()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["counts"]["document_count"] == 1
    assert manifest["schema_version"] == "medical-corpus-fusion-manifest.v2"
    assert manifest["inputs"][0]["documents"]["path"] == "documents.jsonl"
    assert str(tmp_path) not in (output_dir / "manifest.json").read_text()
