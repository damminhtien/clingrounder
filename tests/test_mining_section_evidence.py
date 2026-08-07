"""Source-section evidence tests protect offsets and provenance boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from clingrounder.mining.records import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    MinedDocument,
    RedistributionPolicy,
)
from clingrounder.mining.section_evidence import (
    attach_block_evidence,
    load_block_evidence_policy,
)


def test_attach_block_evidence_preserves_annotations(tmp_path: Path) -> None:
    text = (
        "Case Presentation\n\nPatient had fever.\n\n"
        "Discussion\n\nFever is common."
    )
    raw_blocks = [
        _block(text, "section_title", 0, 17, ["Case Presentation"], "cases"),
        _block(text, "paragraph", 19, 37, ["Case Presentation"], "cases"),
        _block(text, "section_title", 39, 49, ["Discussion"], "discussion"),
        _block(text, "paragraph", 51, 67, ["Discussion"], "discussion"),
    ]
    document = MinedDocument(
        document_id="pmc:fixture",
        text=text,
        language="en",
        note_type="case_report",
        source_artifact_id="pmc:artifact",
        access_class=AccessClass.OPEN,
        redistribution=RedistributionPolicy.ALLOWED,
        hosted_processing_allowed=True,
        metadata={
            "source_block_format": "jats",
            "source_blocks": json.dumps(raw_blocks),
        },
    )
    annotations = (
        _annotation("case-fever", text, (31, 36)),
        _annotation("discussion-fever", text, (51, 56)),
        _annotation("cross-block", text, (15, 21)),
    )
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        """schema_version: medical-block-evidence-policy.v1
policy_id: fixture
source_block_format: jats
default_tier: other
uncontained_tier: uncontained
rules:
  - rule_id: case
    evidence_tier: case_specific
    section_type_patterns: ['^cases$']
  - rule_id: discussion
    evidence_tier: literature_context
    section_type_patterns: ['^discussion$']
""",
        encoding="utf-8",
    )

    result = attach_block_evidence(
        (document,), annotations, load_block_evidence_policy(policy_path)
    )

    by_id = {annotation.annotation_id: annotation for annotation in result.annotations}
    assert by_id["case-fever"].metadata["evidence_tier"] == "case_specific"
    assert by_id["discussion-fever"].metadata["evidence_tier"] == "literature_context"
    assert by_id["cross-block"].metadata["evidence_tier"] == "uncontained"
    assert result.report["span_content_mutation_count"] == 0
    for before in annotations:
        after = by_id[before.annotation_id]
        assert (after.span, after.text, after.entity_type) == (
            before.span,
            before.text,
            before.entity_type,
        )
        assert text[after.span[0] : after.span[1]] == after.text


def _block(
    text: str,
    kind: str,
    start: int,
    end: int,
    section_path: list[str],
    section_type: str,
) -> dict[str, object]:
    return {
        "kind": kind,
        "span": [start, end],
        "section_path": section_path,
        "section_type": section_type,
        "text_sha256": hashlib.sha256(text[start:end].encode("utf-8")).hexdigest(),
    }


def _annotation(
    annotation_id: str,
    text: str,
    span: tuple[int, int],
) -> AnnotationProposal:
    return AnnotationProposal(
        annotation_id=annotation_id,
        document_id="pmc:fixture",
        span=span,
        text=text[span[0] : span[1]],
        entity_type="SYMPTOM",
        assertions=(),
        concepts=(),
        confidence=1.0,
        layer=AnnotationLayer.BRONZE,
        label_source="fixture",
        labeler_id="fixture@1",
    )
