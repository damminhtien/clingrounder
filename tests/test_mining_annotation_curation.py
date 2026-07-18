"""Policy-driven annotation training-view tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import yaml

from medical_kg_nlp.cli.main import main
from medical_kg_nlp.mining.curation import (
    AnnotationCurationPolicy,
    curate_annotations,
)
from medical_kg_nlp.mining.io import write_jsonl
from medical_kg_nlp.mining.records import (
    AnnotationLayer,
    AnnotationProposal,
    ReviewStatus,
)


def _annotation(annotation_id: str, **metadata: str) -> AnnotationProposal:
    return AnnotationProposal(
        annotation_id=annotation_id,
        document_id="doc",
        span=(0, 5),
        text="dolor",
        entity_type="FINDING",
        assertions=(),
        concepts=(),
        confidence=1.0,
        layer=AnnotationLayer.SILVER,
        label_source="source_human_annotation",
        labeler_id="fixture",
        review_status=ReviewStatus.PROPOSED,
        metadata=metadata,
    )


def _policy() -> AnnotationCurationPolicy:
    return AnnotationCurationPolicy(
        policy_id="contiguous-v1",
        allowed_review_statuses=frozenset({ReviewStatus.PROPOSED, ReviewStatus.ACCEPTED}),
        allow_discontinuous=False,
        reject_import_issues=True,
        max_span_length=20,
    )


def test_curation_partitions_without_mutating_source_annotations() -> None:
    accepted = _annotation("accepted", discontinuous="false", import_issues="[]")
    discontinuous = _annotation("discontinuous", discontinuous="true", import_issues="[]")
    source_issue = replace(
        _annotation("source-issue", discontinuous="false", import_issues='["mismatch"]'),
        review_status=ReviewStatus.NEEDS_REVIEW,
    )

    result = curate_annotations(
        (source_issue, accepted, discontinuous),
        _policy(),
    )

    assert result.accepted == (accepted,)
    assert {annotation.annotation_id for annotation in result.rejected} == {
        "discontinuous",
        "source-issue",
    }
    assert result.report["rejection_reason_counts"] == {
        "discontinuous": 1,
        "import_issue": 1,
        "review_status:needs_review": 1,
    }
    assert source_issue.metadata["import_issues"] == '["mismatch"]'


def test_curation_builds_non_overlapping_bio_view_with_audit_winner() -> None:
    short = replace(
        _annotation("short"),
        span=(0, 5),
        text="dolor",
        entity_type="DISEASE",
    )
    long = replace(
        _annotation("long"),
        span=(0, 13),
        text="dolor crónico",
        entity_type="DISEASE",
    )
    separate = replace(
        _annotation("separate"),
        span=(20, 25),
        entity_type="DISEASE",
    )
    policy = replace(
        _policy(),
        allowed_layers=frozenset({AnnotationLayer.SILVER}),
        allowed_entity_types=frozenset({"DISEASE"}),
        overlap_strategy="prefer_quality_longest",
    )

    result = curate_annotations((short, separate, long), policy)

    assert {annotation.annotation_id for annotation in result.accepted} == {
        "long",
        "separate",
    }
    assert [annotation.annotation_id for annotation in result.rejected] == ["short"]
    assert result.report["overlap_winners"] == {"short": "long"}
    assert result.report["rejection_reason_counts"] == {
        "overlap_lower_priority": 1
    }


def test_curation_cli_writes_accepted_rejected_and_report(
    tmp_path: Path,
    capsys,
) -> None:
    annotations = tmp_path / "annotations.jsonl"
    write_jsonl(
        annotations,
        (
            _annotation("accepted", discontinuous="false", import_issues="[]").to_dict(),
            _annotation("rejected", discontinuous="true", import_issues="[]").to_dict(),
        ),
    )
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        yaml.safe_dump(
            {
                "schema_version": "medical-annotation-curation-policy.v1",
                "policy_id": "contiguous-v1",
                "allowed_review_statuses": ["proposed"],
                "allow_discontinuous": False,
                "reject_import_issues": True,
                "max_span_length": 20,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    accepted = tmp_path / "accepted.jsonl"
    rejected = tmp_path / "rejected.jsonl"
    report = tmp_path / "report.json"

    exit_code = main(
        [
            "data",
            "dataset",
            "curate-annotations",
            "--annotations",
            str(annotations),
            "--policy",
            str(policy),
            "--accepted-output",
            str(accepted),
            "--rejected-output",
            str(rejected),
            "--report-output",
            str(report),
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["accepted_count"] == 1
    assert json.loads(accepted.read_text())["annotation_id"] == "accepted"
    assert json.loads(rejected.read_text())["annotation_id"] == "rejected"
    assert json.loads(report.read_text())["policy_id"] == "contiguous-v1"
