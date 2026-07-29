"""Contracts for provenance-separated XLM-R DAPT corpora."""

from __future__ import annotations

import json
from pathlib import Path

from medical_kg_nlp.mining.io import write_json, write_jsonl
from medical_kg_nlp.mining.records import (
    AccessClass,
    MinedDocument,
    RedistributionPolicy,
)
from medical_kg_nlp.training.dapt_corpus import (
    build_dapt_corpus,
    load_dapt_corpus_build_spec,
)
from medical_kg_nlp.cli.parser import build_parser


def test_dapt_corpus_keeps_round2_unlabeled_lane_physically_separate(
    tmp_path: Path,
) -> None:
    open_documents = tmp_path / "open-documents.jsonl"
    round2_documents = tmp_path / "round2-documents.jsonl"
    open_manifest = tmp_path / "open-manifest.json"
    round2_manifest = tmp_path / "round2-manifest.json"
    repeated_text = "Bệnh nhân đau ngực khi gắng sức."
    write_jsonl(
        open_documents,
        (
            _document(
                "open:1",
                repeated_text,
                access_class=AccessClass.OPEN,
                redistribution=RedistributionPolicy.ALLOWED,
                hosted=True,
            ).to_dict(),
        ),
    )
    write_jsonl(
        round2_documents,
        (
            _document(
                "round2:1",
                repeated_text,
                access_class=AccessClass.AUTHORIZED_PRIVATE,
                redistribution=RedistributionPolicy.PROHIBITED,
                hosted=True,
            ).to_dict(),
            _document(
                "round2:2",
                "Câu hỏi về bệnh dại và vết thương ở tay.",
                access_class=AccessClass.AUTHORIZED_PRIVATE,
                redistribution=RedistributionPolicy.PROHIBITED,
                hosted=True,
            ).to_dict(),
        ),
    )
    write_json(open_manifest, {"source": "open", "version": "1"})
    write_json(round2_manifest, {"source": "round2", "version": "1"})
    config = tmp_path / "dapt-corpus.yaml"
    config.write_text(
        f"""
schema_version: xlmr-dapt-corpus-build.v1
build_id: test-dapt-corpus
run_root: .
output_dir: output
lanes:
  - lane_id: vietnamese-open
    kind: open_unlabeled
    documents: {open_documents.name}
    source_manifest: {open_manifest.name}
    sampling_weight: 1.0
  - lane_id: round2-unlabeled
    kind: round2_unlabeled
    documents: {round2_documents.name}
    source_manifest: {round2_manifest.name}
    sampling_weight: 0.25
""".strip()
        + "\n",
        encoding="utf-8",
    )

    manifest = build_dapt_corpus(load_dapt_corpus_build_spec(config))

    assert manifest["record_count"] == 2
    assert manifest["round2_unlabeled_policy"] == {
        "lane_ids": ["round2-unlabeled"],
        "supervision": "none",
        "allowed_objectives": ["masked_language_modeling"],
        "forbidden_objectives": [
            "entity_supervision",
            "pseudo_labeling",
            "synonym_contrastive",
            "threshold_calibration",
        ],
    }
    reports = {row["lane_id"]: row for row in manifest["lanes"]}
    assert reports["vietnamese-open"]["record_count"] == 1
    assert reports["round2-unlabeled"]["record_count"] == 1
    assert (
        reports["round2-unlabeled"]["counters"]["deduplicated.cross_lane"]
        == 1
    )
    round2_row = json.loads(
        (tmp_path / "output/lanes/round2-unlabeled.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert round2_row["document_id"] == "round2:2"
    assert round2_row["supervision"] == "none"
    assert round2_row["objective"] == "masked_language_modeling"
    assert not ({"entities", "annotations", "labels"} & round2_row.keys())


def test_round2_lane_rejects_documents_without_authorized_private_policy(
    tmp_path: Path,
) -> None:
    documents = tmp_path / "documents.jsonl"
    source_manifest = tmp_path / "source-manifest.json"
    write_jsonl(
        documents,
        (
            _document(
                "open:1",
                "Văn bản mở nhưng bị khai báo nhầm là Round 2.",
                access_class=AccessClass.OPEN,
                redistribution=RedistributionPolicy.ALLOWED,
                hosted=True,
            ).to_dict(),
        ),
    )
    write_json(source_manifest, {"source": "open"})
    config = tmp_path / "dapt-corpus.yaml"
    config.write_text(
        f"""
schema_version: xlmr-dapt-corpus-build.v1
build_id: invalid-round2-lane
run_root: .
output_dir: output
lanes:
  - lane_id: round2-unlabeled
    kind: round2_unlabeled
    documents: {documents.name}
    source_manifest: {source_manifest.name}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    manifest = build_dapt_corpus(load_dapt_corpus_build_spec(config))

    assert manifest["record_count"] == 0
    assert manifest["lanes"][0]["counters"]["rejected.round2_access_class"] == 1


def test_dapt_corpus_cli_is_discoverable() -> None:
    args = build_parser().parse_args(
        ["model", "build-dapt-corpus", "--config", "configs/models/dapt.yaml"]
    )

    assert args.handler == "model_build_dapt_corpus"


def _document(
    document_id: str,
    text: str,
    *,
    access_class: AccessClass,
    redistribution: RedistributionPolicy,
    hosted: bool,
) -> MinedDocument:
    return MinedDocument(
        document_id=document_id,
        text=text,
        language="vi",
        note_type="clinical",
        source_artifact_id=f"artifact:{document_id}",
        access_class=access_class,
        redistribution=redistribution,
        hosted_processing_allowed=hosted,
    )
