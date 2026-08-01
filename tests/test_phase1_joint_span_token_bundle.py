"""The joint verifier must preserve token-bundle offsets and source-note OOF groups."""

from __future__ import annotations

from pathlib import Path

from medical_kg_nlp.benchmarks.phase1.joint_span_token_bundle import (
    load_phase1_joint_span_token_bundle,
    prepare_phase1_joint_span_token_bundle,
)
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.mining.io import write_json, write_jsonl
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.utils.hashing import sha256_file, sha256_text


def test_joint_span_token_bundle_keeps_child_offsets_and_parent_group(tmp_path: Path) -> None:
    dataset = tmp_path / "spans.jsonl"
    rows = (_clinical_row(), _qa_row())
    dataset_sha256 = write_jsonl(dataset, rows)
    manifest = tmp_path / "manifest.json"
    write_json(
        manifest,
        {
            "schema_version": "mined-span-dataset.v1",
            "output_sha256": dataset_sha256,
            "chunk_count": len(rows),
            "entity_count": 2,
            "round2_included": False,
            "friend31_included": False,
        },
    )
    build_manifest = tmp_path / "build_manifest.json"
    write_json(
        build_manifest,
        {
            "dataset": {"sha256": dataset_sha256},
            "round2_included": False,
            "friend31_included": False,
        },
    )

    bundle = load_phase1_joint_span_token_bundle(
        dataset_path=dataset,
        manifest_path=manifest,
        build_manifest_path=build_manifest,
    )

    clinical_id = "joint-span-token:clinical-record"
    qa_id = "joint-span-token:qa-record"
    assert bundle.corpus.source_texts[clinical_id] == "đau ngực"
    assert bundle.corpus.source_texts[qa_id] == "Q: đau ngực"
    assert bundle.corpus.gold_rows[clinical_id] == (
        {
            "text": "đau ngực",
            "type": "TRIỆU_CHỨNG",
            "position": [0, 8],
            "assertions": [],
            "candidates": [],
        },
    )
    assert bundle.oof_group_by_document[clinical_id] == "phase1-origin:1"
    assert bundle.oof_group_by_document[qa_id] == "phase1-origin:1"
    assert bundle.manifest["dataset"]["sha256"] == sha256_file(dataset)


def test_joint_span_token_bundle_rejects_unapproved_source(tmp_path: Path) -> None:
    dataset = tmp_path / "spans.jsonl"
    row = _clinical_row()
    row["source_artifact_id"] = "round2-pseudolabels"
    dataset_sha256 = write_jsonl(dataset, (row,))
    manifest = tmp_path / "manifest.json"
    write_json(
        manifest,
        {
            "schema_version": "mined-span-dataset.v1",
            "output_sha256": dataset_sha256,
            "chunk_count": 1,
            "entity_count": 1,
            "round2_included": False,
            "friend31_included": False,
        },
    )

    try:
        load_phase1_joint_span_token_bundle(dataset_path=dataset, manifest_path=manifest)
    except ValueError as error:
        assert "unapproved source artifact" in str(error)
    else:
        raise AssertionError("Round 2 source must be rejected")


def test_joint_span_token_bundle_prepares_bootstrap_lattice(tmp_path: Path) -> None:
    dataset, manifest = _write_bundle(tmp_path)
    bundle = load_phase1_joint_span_token_bundle(dataset_path=dataset, manifest_path=manifest)
    dictionary = DictionaryStore(
        [
            ConceptEntry(
                concept_id="symptom:chest-pain",
                code="LOCAL-CHEST-PAIN",
                code_system=CodeSystem.LOCAL,
                canonical_name="đau ngực",
                semantic_type=EntityType.SYMPTOM,
                aliases=("đau ngực",),
            )
        ]
    )

    report = prepare_phase1_joint_span_token_bundle(
        bundle,
        dictionary,
        output_dir=tmp_path / "prepared",
    )

    assert report["policy"]["model_source_required"] is False
    assert report["supervision"]["oof_group_count"] == 1
    assert Path(report["dataset"]["examples_path"]).is_file()


def _clinical_row() -> dict[str, object]:
    return _row(
        record_id="clinical-record",
        document_id="1",
        text="đau ngực",
        entity_start=0,
        source_artifact_id="phase1-final-supervision:approved",
        note_type="phase1_final_supervision",
        metadata={"source_dataset": "manual_gold"},
    )


def _write_bundle(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "spans.jsonl"
    rows = (_clinical_row(), _qa_row())
    dataset_sha256 = write_jsonl(dataset, rows)
    manifest = tmp_path / "manifest.json"
    write_json(
        manifest,
        {
            "schema_version": "mined-span-dataset.v1",
            "output_sha256": dataset_sha256,
            "chunk_count": len(rows),
            "entity_count": 2,
            "round2_included": False,
            "friend31_included": False,
        },
    )
    return dataset, manifest


def _qa_row() -> dict[str, object]:
    return _row(
        record_id="qa-record",
        document_id="phase1-region-augmentation:example",
        text="Q: đau ngực",
        entity_start=3,
        source_artifact_id="synthetic:phase1-region-renderer.v1",
        note_type="question_answer",
        metadata={"parent_document_id": "phase1-manual-gold:1"},
    )


def _row(
    *,
    record_id: str,
    document_id: str,
    text: str,
    entity_start: int,
    source_artifact_id: str,
    note_type: str,
    metadata: dict[str, str],
) -> dict[str, object]:
    entity_text = "đau ngực"
    entity_end = entity_start + len(entity_text)
    return {
        "record_id": record_id,
        "document_id": document_id,
        "split": "train",
        "text": text,
        "text_sha256": sha256_text(text),
        "source_span": [0, len(text)],
        "language": "vi",
        "note_type": note_type,
        "source_artifact_id": source_artifact_id,
        "metadata": metadata,
        "entities": [
            {
                "annotation_id": f"annotation:{record_id}",
                "start": entity_start,
                "end": entity_end,
                "source_start": entity_start,
                "source_end": entity_end,
                "text": entity_text,
                "label": "SYMPTOM",
            }
        ],
    }
