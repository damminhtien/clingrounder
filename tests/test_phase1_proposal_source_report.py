import json
from pathlib import Path

from medical_kg_nlp.benchmarks.phase1.proposal_source_report import (
    Phase1ProposalSource,
    Phase1SourceSemantics,
    build_phase1_proposal_source_report,
    load_internal_phase1_source,
    write_phase1_proposal_source_report,
)
from medical_kg_nlp.benchmarks.phase1.reviewed_corpus import Phase1ReviewedCorpus
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.output import ClinicalPrediction
from medical_kg_nlp.schema.types import EntityType


def _row(text: str, entity_type: str, start: int, end: int, **extra: object) -> dict:
    return {
        "text": text,
        "type": entity_type,
        "position": [start, end],
        "assertions": [],
        "candidates": [],
        **extra,
    }


def test_source_report_keeps_compatible_labels_distinct_from_targets(
    tmp_path: Path,
) -> None:
    text = "đau ngực và thiếu máu"
    gold = (
        _row("đau ngực", "TRIỆU_CHỨNG", 0, 8),
        _row("thiếu máu", "CHẨN_ĐOÁN", 12, 21),
    )
    corpus = Phase1ReviewedCorpus(
        source_texts={"1": text},
        gold_rows={"1": gold},
        split_by_document={"1": "development"},
    )
    target = Phase1ProposalSource(
        name="rule",
        rows_by_document={
            "1": (
                _row("đau ngực", "CHẨN_ĐOÁN", 0, 8),
                _row("thiếu máu", "CHẨN_ĐOÁN", 12, 21),
            )
        },
    )
    support_rows = []
    for mention, start, end in (("đau ngực", 0, 8), ("thiếu máu", 12, 21)):
        for entity_type in ("CHẨN_ĐOÁN", "TRIỆU_CHỨNG"):
            support_rows.append(
                _row(
                    mention,
                    entity_type,
                    start,
                    end,
                    source_label="DISEASESYMTOM",
                    support_only=True,
                )
            )
    compatible = Phase1ProposalSource(
        name="vietmed",
        rows_by_document={"1": support_rows},
        semantics=Phase1SourceSemantics.COMPATIBLE,
    )

    report = build_phase1_proposal_source_report(
        (target, compatible),
        corpus,
        corpus_fingerprint_sha256="a" * 64,
    )

    rule = report["sources"]["rule"]["splits"]["development"]
    assert rule["exact"]["true_positive"] == 1
    assert rule["error_counts"]["type_confusion"] == 1
    vietmed = report["sources"]["vietmed"]["splits"]["development"]
    assert vietmed["proposal_count"] == 2
    assert vietmed["compatible_exact"]["true_positive"] == 2
    assert vietmed["target_label_note"].startswith("Compatible labels")
    assert report["holdout_opened"] is False

    write_phase1_proposal_source_report(report, tmp_path)
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "errors.jsonl").is_file()
    assert (tmp_path / "summary.md").is_file()


def test_source_report_rejects_partial_split_coverage() -> None:
    corpus = Phase1ReviewedCorpus(
        source_texts={"1": "đau", "2": "sốt"},
        gold_rows={
            "1": (_row("đau", "TRIỆU_CHỨNG", 0, 3),),
            "2": (_row("sốt", "TRIỆU_CHỨNG", 0, 3),),
        },
        split_by_document={"1": "development", "2": "development"},
    )
    source = Phase1ProposalSource(
        name="partial",
        rows_by_document={"1": (_row("đau", "TRIỆU_CHỨNG", 0, 3),)},
    )

    try:
        build_phase1_proposal_source_report(
            (source,),
            corpus,
            corpus_fingerprint_sha256="b" * 64,
        )
    except ValueError as error:
        assert "partially covers development" in str(error)
    else:
        raise AssertionError("Expected partial split coverage to fail")


def test_internal_source_loader_retains_model_confidence(tmp_path: Path) -> None:
    text = "khó thở"
    prediction = ClinicalPrediction.from_text(
        "1",
        text,
        [
            EntityAnnotation(
                id="M1",
                span=(0, 7),
                text=text,
                normalized_text=text,
                type=EntityType.SYMPTOM,
                confidence=0.91,
            )
        ],
        [],
        "test",
    )
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        json.dumps(prediction.to_json(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    rows = load_internal_phase1_source(path, {"1": text})

    assert rows["1"][0]["confidence"] == 0.91
